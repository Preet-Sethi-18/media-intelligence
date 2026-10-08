"""Read-only FastAPI service over the SQLite graph; every error uses one JSON envelope with a request ID."""

import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__, analysis, db
from .config import Settings
from .models import normalize_name, utcnow

logger = logging.getLogger(__name__)

DEFAULT_LIMIT, MAX_LIMIT, MAX_OFFSET = 20, 100, 1_000_000
MAX_NODES, MAX_EDGES, MAX_NAME_LENGTH = 200, 500, 200
MAX_ID = 2**63 - 1
_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

MENTIONS = "Distinct logical source URLs mentioning the entity (document frequency, not span count)"
WEIGHT = "Distinct logical source URLs supporting the typed edge"


class Entity(BaseModel):
    id: str
    name: str
    type: str
    mention_count: int = Field(description=MENTIONS)


class NetworkNode(BaseModel):
    id: str
    name: str
    type: str
    mention_count: int = Field(description=MENTIONS)
    distance: int = Field(description="Minimum hops from the root, following edges in either direction")


class NetworkEdge(BaseModel):
    id: str
    source: str
    target: str
    relation_type: str
    directed: bool
    weight: int = Field(description=WEIGHT)
    first_seen: str
    last_seen: str
    evidence_url: str


class NetworkMeta(BaseModel):
    truncated: bool
    reason: str | None = Field(description="max_nodes, max_edges, max_nodes_and_max_edges, or null")
    max_nodes: int
    max_edges: int


class NetworkResponse(BaseModel):
    root_id: str
    depth: int
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    meta: NetworkMeta


class Thresholds(BaseModel):
    min_delta: int = Field(description="Minimum distinct source URLs first observed in [since, as_of]")
    min_ratio: float = Field(description="Minimum delta / baseline for an existing edge")


class Connection(BaseModel):
    edge_id: str
    source: Entity
    target: Entity
    relation_type: str
    directed: bool
    first_seen: str
    last_seen: str
    reason: Literal["new", "grown"]
    weight_before: int = Field(description="Distinct URLs first observed before since")
    weight_now: int = Field(description="weight_before + weight_delta, as of the snapshot")
    weight_delta: int = Field(description="Distinct URLs first observed from since through as_of")
    relative_growth: float | None = Field(description="weight_delta / weight_before; null for a new edge")
    latest_contribution: str = Field(description="Latest first observation counted in weight_delta")
    evidence_url: str


class ConnectionsResponse(BaseModel):
    since: str
    as_of: str
    thresholds: Thresholds
    items: list[Connection]
    total: int
    limit: int
    offset: int


class CentralEntity(BaseModel):
    rank: int
    node_id: str
    name: str
    type: str
    score: float
    degree: int = Field(description="Distinct neighbours, ignoring direction and relation type")
    relation_type_count: int
    weighted_degree: int = Field(description="Sum of incident edge weights")
    mention_count: int = Field(description=MENTIONS)


class CentralResponse(BaseModel):
    metric: Literal["normalized_degree"]
    definition: str
    node_count: int
    items: list[CentralEntity]
    total: int
    limit: int
    offset: int


class EdgeSummary(BaseModel):
    id: str
    source: Entity
    target: Entity
    relation_type: str
    directed: bool
    weight: int = Field(description=WEIGHT)
    first_seen: str
    last_seen: str


class Evidence(BaseModel):
    evidence_id: str
    source_id: str
    source_url: str
    source_type: str
    title: str | None
    author: str | None
    published_at: str | None
    scraped_at: str
    last_scraped_at: str
    observed_at: str
    evidence_text: str
    segment_id: str
    segment_kind: str | None = Field(description="body, title, post, comment, reply or context")
    segment_author: str | None = Field(description="Author of this segment (a comment's own author, not the "
                                                   "thread submitter); null when unavailable")
    segment_url: str | None = Field(description="Permalink of the comment or reply when the source exposes one")
    segment_published_at: str | None = Field(description="Publication time of this segment when available")
    start_offset: int
    end_offset: int
    rule_id: str
    quality_tier: str


class EvidenceResponse(BaseModel):
    edge: EdgeSummary
    items: list[Evidence]
    total: int = Field(description="Evidence versions; can exceed weight when one URL has several versions")
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    schema_version: int
    counts: dict[str, int]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None
    candidates: list[Entity] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def _invalid(field: str, message: str, kind: str) -> ApiError:
    return ApiError(422, "validation_error", "Request validation failed",
                    details=[{"loc": ["query", field], "msg": message, "type": kind}])


def _request_id(request: Request) -> str:
    if not getattr(request.state, "request_id", None):
        request.state.request_id = uuid4().hex
    return request.state.request_id


def _error(request: Request, status: int, code: str, message: str, headers: dict | None = None,
           **extra: Any) -> JSONResponse:
    request_id = _request_id(request)
    body = {"code": code, "message": message, "request_id": request_id}
    body.update({key: value for key, value in extra.items() if value is not None})
    return JSONResponse({"error": body}, status_code=status, headers={**(headers or {}), "X-Request-ID": request_id})


def _errors(*statuses: int) -> dict[int | str, dict[str, Any]]:
    return {status: {"model": ErrorResponse} for status in statuses}


_DECODED_PLUS = re.compile(r"(T[0-9:.,]+) ([0-9]{2}(?::?[0-9]{2})?)$")


def _parse_since(value: str) -> datetime:
    # An unencoded "+" in a query string decodes to a space: "...T12:00:00 00:00" was typed as "+00:00".
    value = _DECODED_PLUS.sub(r"\1+\2", value.strip())
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise _invalid("since", "Expected an ISO 8601 timestamp such as 2026-10-07T12:00:00Z; "
                                "percent-encode '+' in offsets as %2B", "datetime_parsing") from None
    if parsed.utcoffset() is None:
        raise _invalid("since", "since must include a timezone (Z or +HH:MM)", "timezone_required")
    try:
        return parsed.astimezone(UTC)
    except OverflowError:
        raise _invalid("since", "since is outside the supported date range", "datetime_range") from None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title="Media Intelligence Graph API", version=__version__,
                  description="Read-only queries over the evidence-backed SQLite knowledge graph.")
    app.state.settings = settings

    @contextmanager
    def reader() -> Iterator[sqlite3.Connection]:
        # Per-request read-only connection: the API never creates, migrates, or writes the database.
        conn = db.connect(settings.database_path, readonly=True)
        try:
            db.check_schema(conn)
            yield conn
        finally:
            conn.close()

    @app.middleware("http")
    async def request_id_header(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if _REQUEST_ID.fullmatch(incoming) else uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError):
        return _error(request, exc.status, exc.code, exc.message, **exc.extra)

    @app.exception_handler(analysis.EntityNotFound)
    async def entity_not_found(request: Request, exc: analysis.EntityNotFound):
        return _error(request, 404, "entity_not_found", str(exc))

    @app.exception_handler(analysis.AmbiguousEntity)
    async def ambiguous_entity(request: Request, exc: analysis.AmbiguousEntity):
        return _error(request, 409, "ambiguous_entity", str(exc), candidates=exc.candidates)

    @app.exception_handler(analysis.EdgeNotFound)
    async def edge_not_found(request: Request, exc: analysis.EdgeNotFound):
        return _error(request, 404, "edge_not_found", str(exc))

    @app.exception_handler(db.SchemaError)
    async def database_unavailable(request: Request, exc: db.SchemaError):
        # The detail can include a local file path, so it is logged rather than returned.
        logger.warning("Database unavailable (request %s): %s", _request_id(request), exc)
        return _error(request, 503, "database_unavailable",
                      "The graph database is missing or not migrated; run the ingest command first")

    @app.exception_handler(sqlite3.OperationalError)
    async def database_error(request: Request, exc: sqlite3.OperationalError):
        if any(word in str(exc).lower() for word in ("locked", "busy")):
            logger.warning("Database busy (request %s): %s", _request_id(request), exc)
            return _error(request, 503, "database_busy", "The graph database is busy; retry shortly")
        return await internal_error(request, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        details = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
        return _error(request, 422, "validation_error", "Request validation failed", details=details)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error(request, exc.status_code, code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        logger.error("Unexpected error (request %s)", _request_id(request), exc_info=exc)
        return _error(request, 500, "internal_error", "An unexpected internal error occurred")

    @app.get("/entity/{name:path}/network", response_model=NetworkResponse, responses=_errors(404, 409, 422, 503))
    def entity_network(
        name: Annotated[str, Path(min_length=1, max_length=MAX_NAME_LENGTH,
                                  description="Canonical name or alias; percent-encode spaces and '@'")],
        depth: Annotated[int, Query(ge=1, le=2)] = 2,
        entity_id: Annotated[int | None, Query(ge=1, le=MAX_ID, description="Choose one ambiguous match")] = None,
        max_nodes: Annotated[int, Query(ge=1, le=MAX_NODES)] = MAX_NODES,
        max_edges: Annotated[int, Query(ge=1, le=MAX_EDGES)] = MAX_EDGES,
    ):
        """Depth 1 returns direct neighbours; depth 2 adds their neighbours. Edges keep their stored direction."""
        if not normalize_name(name):
            raise ApiError(422, "validation_error", "Request validation failed",
                           details=[{"loc": ["path", "name"], "msg": "Name must not be blank", "type": "blank"}])
        with reader() as conn:
            return analysis.network(conn, name, depth=depth, entity_id=entity_id, max_nodes=max_nodes,
                                    max_edges=max_edges)

    @app.get("/connections/new", response_model=ConnectionsResponse, responses=_errors(422, 503))
    def new_connections(
        since: Annotated[str, Query(min_length=1, max_length=64,
                                    description="Timezone-aware ISO 8601 cutoff, e.g. 2026-10-07T12:00:00Z")],
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
        offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    ):
        """New edges, and existing edges whose distinct-source support grew past both configured thresholds."""
        cutoff = _parse_since(since)
        as_of = utcnow()
        if cutoff > as_of:
            raise _invalid("since", "since must not be later than the current time (as_of)", "datetime_future")
        with reader() as conn:
            return analysis.new_connections(conn, cutoff, as_of=as_of, min_delta=settings.growth_min_delta,
                                            min_ratio=settings.growth_min_ratio, limit=limit, offset=offset)

    @app.get("/entities/central", response_model=CentralResponse, responses=_errors(422, 503))
    def central_entities(
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
        offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    ):
        """Entities ranked by normalized degree centrality; component counts explain each score."""
        with reader() as conn:
            return analysis.central(conn, limit=limit, offset=offset)

    @app.get("/edges/{edge_id}/sources", response_model=EvidenceResponse, responses=_errors(404, 422, 503))
    def edge_sources(
        edge_id: Annotated[int, Path(ge=1, le=MAX_ID)],
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
        offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
    ):
        """Every stored source version supporting the edge, with its exact evidence excerpt."""
        with reader() as conn:
            return analysis.edge_sources(conn, edge_id, limit=limit, offset=offset)

    @app.get("/health", response_model=HealthResponse, responses=_errors(503))
    def health():
        """Database and schema readiness plus the application version."""
        with reader() as conn:
            return {"status": "ok", "version": __version__, "schema_version": db.user_version(conn),
                    "counts": analysis.graph_counts(conn)}

    return app


app = create_app()
