"""SQLite connections, explicit transactions, and numbered schema migrations tracked by PRAGMA user_version."""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Every stored time is models.timestamp() output, so plain string comparison orders it correctly.
_TS = ("GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]"
       ".[0-9][0-9][0-9][0-9][0-9][0-9]Z'")

_INITIAL = f"""
CREATE TABLE crawl_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL CHECK (started_at {_TS}),
    finished_at TEXT CHECK (finished_at {_TS}),
    status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'complete', 'partial', 'failed')),
    config_hash TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    summary_json TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(summary_json)),
    CHECK (finished_at >= started_at),
    CHECK (status = 'running' OR finished_at IS NOT NULL)
) STRICT;

CREATE TABLE sources (
    id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL CHECK (source_url <> ''),
    requested_url TEXT NOT NULL CHECK (requested_url <> ''),
    source_type TEXT NOT NULL CHECK (source_type IN ('news', 'discussion', 'microblog', 'official', 'analysis')),
    scraped_at TEXT NOT NULL CHECK (scraped_at {_TS}),
    last_scraped_at TEXT NOT NULL CHECK (last_scraped_at {_TS}),
    title TEXT,
    body TEXT NOT NULL CHECK (body <> ''),
    author TEXT,
    published_at TEXT CHECK (published_at {_TS}),
    content_hash TEXT NOT NULL CHECK (content_hash <> ''),
    segments_json TEXT NOT NULL DEFAULT '[]' CHECK (json_valid(segments_json) AND json_type(segments_json) = 'array'),
    metadata_json TEXT NOT NULL DEFAULT '{{}}' CHECK (json_valid(metadata_json) AND json_type(metadata_json) = 'object'),
    crawl_run_id INTEGER NOT NULL REFERENCES crawl_runs (id),
    UNIQUE (source_url, content_hash),
    CHECK (last_scraped_at >= scraped_at)
) STRICT;
CREATE INDEX sources_content_hash ON sources (content_hash);
CREATE INDEX sources_crawl_run ON sources (crawl_run_id);

CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL CHECK (trim(name) <> ''),
    normalized_name TEXT NOT NULL CHECK (normalized_name <> ''),
    entity_type TEXT NOT NULL CHECK (entity_type IN ('PERSON', 'ORG', 'LOCATION', 'TOPIC')),
    first_seen TEXT NOT NULL CHECK (first_seen {_TS}),
    mention_count INTEGER NOT NULL DEFAULT 0 CHECK (mention_count >= 0),
    UNIQUE (entity_type, normalized_name),
    CHECK (key = entity_type || ':' || normalized_name)
) STRICT;
CREATE INDEX nodes_normalized_name ON nodes (normalized_name);

-- Source-endpoint lookups use the UNIQUE index, whose leading column is source_node_id.
CREATE TABLE edges (
    id INTEGER PRIMARY KEY,
    source_node_id INTEGER NOT NULL REFERENCES nodes (id),
    target_node_id INTEGER NOT NULL REFERENCES nodes (id),
    relation_type TEXT NOT NULL CHECK (relation_type <> ''),
    directed INTEGER NOT NULL CHECK (directed IN (0, 1)),
    weight INTEGER NOT NULL CHECK (weight >= 1),
    first_seen TEXT NOT NULL CHECK (first_seen {_TS}),
    last_seen TEXT NOT NULL CHECK (last_seen {_TS}),
    UNIQUE (source_node_id, target_node_id, relation_type),
    CHECK (source_node_id <> target_node_id),
    CHECK (directed = 1 OR source_node_id < target_node_id),
    CHECK (last_seen >= first_seen)
) STRICT;
CREATE INDEX edges_target_node ON edges (target_node_id);
CREATE INDEX edges_relation_type ON edges (relation_type);

-- A surrogate id keeps evidence references stable for API pagination (implicit rowids can change on VACUUM).
CREATE TABLE edge_sources (
    id INTEGER PRIMARY KEY,
    edge_id INTEGER NOT NULL REFERENCES edges (id),
    source_id INTEGER NOT NULL REFERENCES sources (id),
    observed_at TEXT NOT NULL CHECK (observed_at {_TS}),
    evidence_text TEXT NOT NULL CHECK (evidence_text <> ''),
    segment_id TEXT NOT NULL CHECK (segment_id <> ''),
    start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
    end_offset INTEGER NOT NULL CHECK (end_offset >= start_offset),
    rule_id TEXT NOT NULL CHECK (rule_id <> ''),
    quality_tier TEXT NOT NULL CHECK (quality_tier IN ('semantic_rule', 'cooccurrence')),
    UNIQUE (edge_id, source_id)
) STRICT;
CREATE INDEX edge_sources_edge_observed ON edge_sources (edge_id, observed_at);
CREATE INDEX edge_sources_source ON edge_sources (source_id);

CREATE TABLE node_mentions (
    node_id INTEGER NOT NULL REFERENCES nodes (id),
    source_id INTEGER NOT NULL REFERENCES sources (id),
    first_observed_at TEXT NOT NULL CHECK (first_observed_at {_TS}),
    surface_forms_json TEXT NOT NULL DEFAULT '[]'
        CHECK (json_valid(surface_forms_json) AND json_type(surface_forms_json) = 'array'),
    occurrence_count INTEGER NOT NULL CHECK (occurrence_count >= 1),
    PRIMARY KEY (node_id, source_id)
) STRICT;
CREATE INDEX node_mentions_source ON node_mentions (source_id);

-- One alias key may name several nodes (ambiguity), so uniqueness is per (alias_key, node_id) only.
CREATE TABLE node_aliases (
    alias_key TEXT NOT NULL CHECK (alias_key <> ''),
    node_id INTEGER NOT NULL REFERENCES nodes (id),
    alias_kind TEXT NOT NULL CHECK (alias_kind <> ''),
    origin TEXT NOT NULL CHECK (origin <> ''),
    PRIMARY KEY (alias_key, node_id)
) STRICT;
CREATE INDEX node_aliases_node ON node_aliases (node_id);
"""

# Append-only: never edit a released migration; add a new one instead.
MIGRATIONS: tuple[str, ...] = (_INITIAL,)
SCHEMA_VERSION: int = len(MIGRATIONS)


class SchemaError(RuntimeError):
    """The database is missing, unreadable, or not at the schema version this code expects."""


def connect(path: str | Path, *, readonly: bool = False, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced; readonly opens an existing file with mode=ro."""
    path = Path(path)
    memory = str(path) == ":memory:"
    if readonly:
        try:
            if memory or not path.is_file():
                raise SchemaError(f"Database file not found: {path}")
            # A serialized SQLite build (threadsafety 3) lets a read connection move between threadpool threads.
            conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None,
                                   timeout=busy_timeout_ms / 1000, check_same_thread=sqlite3.threadsafety != 3)
        except (OSError, sqlite3.Error) as exc:  # e.g. no read permission on the file or its directory
            raise SchemaError(f"Cannot open database {path}: {exc}") from exc
    else:
        if not memory:
            path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, isolation_level=None, timeout=busy_timeout_ms / 1000)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys = ON")
        if not readonly and not memory:
            mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if mode.lower() != "wal":
                logger.warning("WAL journal mode unavailable for %s; using %s", path, mode)
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise SchemaError(f"Cannot open database {path}: {exc}") from exc
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN IMMEDIATE ... COMMIT; any exception rolls back and propagates. Not reentrant."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        # SQLite may already have rolled back on some errors; a second ROLLBACK would mask the cause.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _statements(script: str) -> Iterator[str]:
    # Connection.executescript() commits any open transaction first, so statements run one by one instead.
    buffer = ""
    for part in script.split(";"):
        buffer += part + ";"
        if sqlite3.complete_statement(buffer):
            if statement := buffer.strip().removesuffix(";").strip():
                yield statement
            buffer = ""
    if buffer.removesuffix(";").strip():
        raise SchemaError("Migration script ends with an incomplete statement")


def migrate(conn: sqlite3.Connection) -> None:
    """Apply pending migrations, one transaction each; refuse a database newer than this code."""
    while True:
        with transaction(conn):
            # Read inside the write lock so a concurrent migrator cannot apply the same step twice.
            current = user_version(conn)
            if current > SCHEMA_VERSION:
                raise SchemaError(f"Database schema version {current} is newer than supported {SCHEMA_VERSION}")
            if current == SCHEMA_VERSION:
                return
            for statement in _statements(MIGRATIONS[current]):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {current + 1}")
        logger.info("Applied database migration %d", current + 1)


def check_schema(conn: sqlite3.Connection) -> None:
    """Raise SchemaError unless the database is fully migrated; the API maps this to HTTP 503."""
    try:
        current = user_version(conn)
    except sqlite3.DatabaseError as exc:
        raise SchemaError(f"Database is unreadable: {exc}") from exc
    if current != SCHEMA_VERSION:
        raise SchemaError(f"Database schema version is {current}; expected {SCHEMA_VERSION}. Run ingest to migrate.")


_INTEGRITY_CHECKS = {
    "edges_without_evidence": """SELECT count(*) FROM edges e
        WHERE NOT EXISTS (SELECT 1 FROM edge_sources es WHERE es.edge_id = e.id)""",
    "edge_weight_mismatches": """SELECT count(*) FROM edges e WHERE e.weight <> (
        SELECT count(DISTINCT s.source_url) FROM edge_sources es JOIN sources s ON s.id = es.source_id
        WHERE es.edge_id = e.id)""",
    "edge_first_seen_mismatches": """SELECT count(*) FROM edges e
        WHERE e.first_seen <> (SELECT min(observed_at) FROM edge_sources es WHERE es.edge_id = e.id)""",
    "node_mention_count_mismatches": """SELECT count(*) FROM nodes n WHERE n.mention_count <> (
        SELECT count(DISTINCT s.source_url) FROM node_mentions nm JOIN sources s ON s.id = nm.source_id
        WHERE nm.node_id = n.id)""",
    "node_first_seen_mismatches": """SELECT count(*) FROM nodes n
        WHERE n.first_seen <> (SELECT min(first_observed_at) FROM node_mentions nm WHERE nm.node_id = n.id)""",
}


def integrity_report(conn: sqlite3.Connection) -> dict[str, int]:
    """Count violations of invariants SQLite cannot express as constraints; all zeros means consistent.

    Time mismatches are only counted where evidence exists, so an evidence-less edge is reported once per cause.
    """
    report = {"foreign_key_violations": len(conn.execute("PRAGMA foreign_key_check").fetchall())}
    report.update({name: conn.execute(sql).fetchone()[0] for name, sql in _INTEGRITY_CHECKS.items()})
    return report
