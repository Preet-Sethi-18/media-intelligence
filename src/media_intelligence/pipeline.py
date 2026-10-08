"""Ingest orchestration: crawl, normalize, extract, and store each item in its own transaction, then report."""

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .config import PipelineConfig, Settings
from .crawler import BrowserCrawler
from .db import SchemaError, connect, migrate
from .extraction import Extractor
from .models import ContentItem, Extraction, FetchResult, timestamp, utcnow
from .normalization import NormalizationError, normalize
from .storage import finish_run, graph_counts, source_version_id, start_run, store_item, verify_integrity

logger = logging.getLogger(__name__)

EXIT_CODES = {"complete": 0, "failed": 1, "partial": 3}
BLOCKED_STATUSES = frozenset({401, 403, 429})
MAX_CONSECUTIVE_DB_FAILURES = 2
OUTCOMES = ("failed", "blocked", "skipped", "extraction_failed", "storage_failed", "stored_new", "stored_unchanged")
STORE_FIELDS = ("nodes_created", "edges_created", "evidence_added", "mentions_added", "relations_skipped",
                "self_edges_rejected", "invalid_evidence")
_URL_TAIL = re.compile(r"(https?://[^\s?#'\"<>]*)[?#][^\s'\"<>]*")


def safe_url(url: str) -> str:
    """Scheme, host, port, and path only: queries, fragments, and credentials never reach logs or reports."""
    try:
        parts = urlsplit(url)
        netloc = (parts.hostname or "") + (f":{parts.port}" if parts.port else "")
    except ValueError:
        return "<invalid url>"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _scrub(text: str, limit: int = 120) -> str:
    return " ".join(_URL_TAIL.sub(r"\1", text).split())[:limit]


def _code(message: str) -> str:
    # Normalization errors are "<code>: <explanation>"; the code is the stable, countable part.
    if message.startswith("Unexpected error in _crawl_web"):
        # Crawl4AI wraps browser failures in a long traceback-like message; the last line names the cause.
        cause = message.strip().splitlines()[-1]
        return "crawl_failed: " + _scrub(re.sub(r"https?://\S+", "<url>", cause), 80)
    return _scrub(message.partition(":")[0], 60) or "unknown"


def _warning_kind(message: str) -> str:
    # Drop per-item detail (quoted names, raw dates, counts) so warnings aggregate into a few kinds.
    return " ".join(re.split(r"[:'\"]| with \d| for ", message, maxsplit=1)[0].split())[:100]


class _Skip(Exception):
    """An item that ends without a stored version; the run continues."""

    def __init__(self, outcome: str, reason: str):
        super().__init__(reason)
        self.outcome, self.reason = outcome, reason


class _Fatal(_Skip):
    """A persistent database failure: the item is counted and the run stops."""


@dataclass
class SourceTally:
    attempted: int = 0
    fetched: int = 0
    failed: int = 0
    blocked: int = 0
    skipped: int = 0
    extraction_failed: int = 0
    storage_failed: int = 0
    stored_new: int = 0
    stored_unchanged: int = 0
    segments: int = 0
    truncated: int = 0
    missing_metadata: Counter = field(default_factory=lambda: Counter(title=0, author=0, published_at=0))
    reasons: defaultdict = field(default_factory=lambda: defaultdict(Counter))

    @property
    def stored(self) -> int:
        return self.stored_new + self.stored_unchanged

    def record(self, outcome: str, reason: str | None = None) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)
        if reason:
            self.reasons[outcome][reason] += 1

    def as_dict(self) -> dict:
        counts = {name: getattr(self, name) for name in ("attempted", "fetched", *OUTCOMES, "segments", "truncated")}
        return {**counts, "stored": self.stored, "missing_metadata": dict(self.missing_metadata),
                "reasons": {outcome: dict(sorted(c.items())) for outcome, c in sorted(self.reasons.items())}}


def _report_folder(root: Path, started: datetime, run_id: int) -> Path:
    # The run id disambiguates runs started in the same second; the suffix covers two databases sharing a folder.
    base = f"run-{started:%Y%m%dT%H%M%SZ}-{run_id}"
    for attempt in count():
        folder = root / (base if attempt == 0 else f"{base}.{attempt}")
        try:
            folder.mkdir(parents=True)
            return folder
        except FileExistsError:
            continue


def _load_extractor(model: str, aliases: Path, topics: Path) -> Extractor:
    try:
        return Extractor(model, aliases, topics)
    except OSError as exc:
        raise ValueError(f"spaCy model {model!r} could not be loaded; install it with uv sync "
                         f"or set MI_SPACY_MODEL to an installed model ({exc})") from exc


class _IngestRun:
    def __init__(self, config: PipelineConfig, settings: Settings, conn: sqlite3.Connection, extractor: Extractor,
                 config_path: Path, config_hash: str, *, live: bool):
        self.config, self.settings, self.conn, self.extractor = config, settings, conn, extractor
        self.config_path, self.config_hash, self.live = config_path, config_hash, live
        self.profiles = {source.name: source for source in config.sources}
        self.tallies = {name: SourceTally() for name in self.profiles}
        self.warnings = {"normalization": Counter(), "extraction": Counter()}
        self.storage = Counter(dict.fromkeys(STORE_FIELDS, 0))
        self.db_failures = 0
        self.seed_failures: Counter = Counter()
        self.run_id = 0
        self.started = utcnow()

    async def execute(self, factory: Callable[[PipelineConfig, Settings], Any]) -> int:
        # Constructing the crawler validates its setup (e.g. the browser state file) before a run row exists, so a
        # setup error is reported as one (exit 2) instead of as a recorded failed run.
        crawler = factory(self.config, self.settings)
        self.run_id = start_run(self.conn, config_hash=self.config_hash, extractor_version=self.extractor.version,
                                started_at=self.started)
        logger.info("run_started run_id=%d live=%s config_hash=%s extractor=%s",
                    self.run_id, self.live, self.config_hash[:12], self.extractor.version)
        fatal, interrupt = None, None
        try:
            async with aclosing(crawler.pages()) as pages:
                async for page in pages:
                    await self._process(page)
        except _Fatal as exc:
            fatal = exc.reason
        except Exception as exc:  # a crawler crash is a recorded failed run with its own exit code
            fatal = _scrub(f"crawl_aborted: {type(exc).__name__}: {exc}", 200)
            logger.error("run_id=%d %s", self.run_id, fatal)  # scrubbed: no URL queries or page text in logs
            logger.debug("crawler traceback", exc_info=exc)
        except BaseException as exc:  # noqa: BLE001 - interrupts and cancellation are recorded, then re-raised
            fatal, interrupt = f"crawl_interrupted: {type(exc).__name__}", exc
        code = self._finish(crawler, fatal)
        if interrupt is not None:
            raise interrupt
        return code

    async def _process(self, page: FetchResult) -> None:
        clock = time.monotonic()
        tally = self.tallies.setdefault(page.source_name, SourceTally())
        tally.attempted += 1
        outcome, reason, stop = "", None, None
        try:
            outcome = await self._handle(page, tally)
        except _Skip as skip:
            outcome, reason = skip.outcome, skip.reason
            stop = skip if isinstance(skip, _Fatal) else None
        tally.record(outcome, reason)
        # A configured listing seed is only crawled for its links, so its skip is expected, not a failure.
        if page.depth == 0 and not outcome.startswith("stored") and reason != "listing_page":
            self.seed_failures[page.source_name] += 1
        logger.info("item run_id=%d source=%s depth=%d outcome=%s reason=%s elapsed_ms=%d url=%s",
                    self.run_id, page.source_name, page.depth, outcome, reason, (time.monotonic() - clock) * 1000,
                    safe_url(page.final_url or page.requested_url))
        if stop is not None:
            raise stop

    async def _handle(self, page: FetchResult, tally: SourceTally) -> str:
        if not page.success or (page.status_code or 0) >= 400:
            if page.status_code in BLOCKED_STATUSES:
                raise _Skip("blocked", f"http_{page.status_code}")
            raise _Skip("failed", f"http_{page.status_code}" if page.status_code else _code(page.error or "unknown"))
        tally.fetched += 1
        # The profile that queued the page wins while it still accepts the final URL (two profiles may share a host).
        declared = self.profiles.get(page.source_name)
        profile = declared if declared is not None and declared.accepts(page.final_url) else (
            self.config.source_for(page.final_url) or declared)
        if profile is None:
            raise _Skip("skipped", "outside_source_policy")
        try:
            item = normalize(page, profile, self.config.crawl)
        except NormalizationError as exc:
            code = _code(str(exc))
            # Login walls and challenge pages are access failures, not empty content.
            raise _Skip("blocked" if code.startswith("access") else "skipped", code) from None
        except Exception as exc:  # noqa: BLE001 - one malformed page must not end the run
            raise _Skip("skipped", f"normalization_error: {type(exc).__name__}") from None
        self._observe(item, tally)
        extraction = Extraction()
        # An already stored (URL, content hash) version only refreshes recency, so spaCy is skipped for it.
        if self._database(source_version_id, self.conn, item) is None:
            try:
                extraction = await asyncio.to_thread(self.extractor.extract, item)
            except Exception as exc:  # noqa: BLE001 - an NLP failure skips the item; nothing was written
                logger.warning("extraction_failed url=%s error=%s", safe_url(item.source_url), type(exc).__name__)
                raise _Skip("extraction_failed", f"extraction_error: {type(exc).__name__}") from None
            self.warnings["extraction"].update(_warning_kind(w) for w in extraction.warnings)
        # Evidence is stamped with the store-time clock inside the write transaction (see store_item).
        result = self._database(store_item, self.conn, item, extraction, run_id=self.run_id, write=True,
                                reviewed_aliases=getattr(self.extractor, "reviewed", None))
        self.storage.update({name: getattr(result, name) for name in STORE_FIELDS})
        return "stored_new" if result.new_version else "stored_unchanged"

    def _observe(self, item: ContentItem, tally: SourceTally) -> None:
        tally.segments += len(item.segments)
        tally.truncated += bool(item.metadata.get("truncated"))
        for name in ("title", "author", "published_at"):
            tally.missing_metadata[name] += getattr(item, name) is None
        self.warnings["normalization"].update(_warning_kind(w) for w in item.metadata.get("warnings", []))

    def _database(self, operation: Callable, *args, write: bool = False, **kwargs):
        """Run one database operation. Only a write that the database answers (success or an item-specific
        rejection) ends a failure streak; a successful read does not, so persistent write failures still stop."""
        try:
            result = operation(*args, **kwargs)
        except sqlite3.IntegrityError as exc:
            # A constraint violation is specific to this item's data; the transaction was rolled back.
            if write:
                self.db_failures = 0
            logger.warning("storage_rejected error=%s", _scrub(str(exc)))
            raise _Skip("storage_failed", "integrity_error") from None
        except sqlite3.DatabaseError as exc:
            self.db_failures += 1
            logger.error("database_error consecutive=%d error=%s: %s", self.db_failures, type(exc).__name__,
                         _scrub(str(exc)))
            reason = f"database_error: {type(exc).__name__}"
            if self.db_failures >= MAX_CONSECUTIVE_DB_FAILURES:
                raise _Fatal("storage_failed", reason) from None
            raise _Skip("storage_failed", reason) from None
        except Exception as exc:  # noqa: BLE001 - e.g. inconsistent extractor output rejected before writing
            if write:
                self.db_failures = 0
            logger.warning("storage_rejected error=%s: %s", type(exc).__name__, _scrub(str(exc)))
            raise _Skip("storage_failed", f"storage_rejected: {type(exc).__name__}") from None
        if write:
            self.db_failures = 0
        return result

    def _finish(self, crawler: Any, fatal: str | None) -> int:
        finished = utcnow()
        graph, problems = None, None
        try:
            graph, problems = graph_counts(self.conn), verify_integrity(self.conn)
        except sqlite3.Error as exc:
            fatal = fatal or f"database_error: {type(exc).__name__}"
            logger.error("final graph checks failed: %s", _scrub(str(exc)))
        missing = [name for name in self.profiles if self.tallies[name].stored == 0]
        failed_seeds = dict(sorted(self.seed_failures.items()))
        status = "failed" if fatal else "partial" if missing or problems or failed_seeds else "complete"
        tiers = (graph or {}).get("edges_by_quality_tier", {})
        quality = {"semantic_rule": tiers.get("semantic_rule", 0), "cooccurrence": tiers.get("cooccurrence", 0)}
        warnings = []
        if fatal:
            warnings.append(f"Run aborted: {fatal}")
        if missing:
            warnings.append(f"Required sources without a stored item: {', '.join(missing)}")
        if failed_seeds:
            counts = ", ".join(f"{name}={count}" for name, count in failed_seeds.items())
            warnings.append(f"Seed pages without a stored item (see sources.reasons): {counts}")
        if problems:
            warnings.append(f"Integrity check failed: {'; '.join(problems)}")
        if graph is not None and not graph.get("edges"):
            warnings.append("The graph has no edges")
        elif graph is not None and not quality["semantic_rule"]:
            warnings.append("No semantic_rule edges exist; the graph has only co-occurrence links")
        if self.storage["invalid_evidence"]:
            warnings.append(f"{self.storage['invalid_evidence']} extracted relations had invalid evidence offsets")
        report_path: Path | None = None
        try:
            report_path = _report_folder(self.settings.report_dir, self.started, self.run_id) / "report.json"
        except OSError as exc:  # the run status is still recorded below
            logger.error("report folder could not be created: %s", _scrub(str(exc)))
            warnings.append(f"Run report could not be written: {type(exc).__name__}")
        totals, reasons = Counter(), defaultdict(Counter)
        for tally in self.tallies.values():
            totals.update({name: getattr(tally, name) for name in ("attempted", "fetched", *OUTCOMES, "truncated")})
            for outcome, counter in tally.reasons.items():
                reasons[outcome].update(counter)

        def summary() -> dict:
            return {"run_id": self.run_id, "status": status, "exit_code": EXIT_CODES[status], "live": self.live,
                    "stored_new": totals["stored_new"], "stored_unchanged": totals["stored_unchanged"],
                    "missing_sources": missing, "failed_seeds": failed_seeds, "nodes": (graph or {}).get("nodes"),
                    "edges": (graph or {}).get("edges"), "semantic_edges": quality["semantic_rule"],
                    "integrity_ok": problems == [], "warnings": warnings,
                    "report": str(report_path) if report_path else None}

        def record() -> None:
            nonlocal status
            try:
                finish_run(self.conn, self.run_id, status=status, summary=summary(), finished_at=finished)
            except sqlite3.Error as exc:
                status = "failed"
                warnings.append(f"Run status could not be recorded in the database: {type(exc).__name__}")

        record()
        frontier = getattr(crawler, "frontier", None)
        report = {
            **summary(), "kind": "ingest_run",
            "evidence_note": ("live Crawl4AI crawl" if self.live
                              else "offline run with an injected crawler; not live crawl evidence"),
            "started_at": timestamp(self.started), "finished_at": timestamp(finished),
            "config_path": str(self.config_path), "config_hash": self.config_hash,
            "extractor_version": self.extractor.version, "crawl_limits": self.config.crawl.model_dump(),
            "fatal_error": fatal, "totals": dict(totals),
            "sources": {name: tally.as_dict() for name, tally in self.tallies.items()},
            "skip_reasons": {outcome: dict(sorted(c.items())) for outcome, c in sorted(reasons.items())},
            "crawl_skips": dict(sorted(getattr(frontier, "skips", {}).items())),
            "item_warnings": {stage: dict(c.most_common()) for stage, c in self.warnings.items()},
            "storage": dict(self.storage), "graph": graph, "edge_quality": quality,
            "integrity": {"ok": problems == [], "problems": problems},
        }
        if report_path is not None:
            try:
                report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.error("report could not be written: %s", _scrub(str(exc)))
                report_path = None
                warnings.append(f"Run report could not be written: {type(exc).__name__}")
                record()  # the stored summary must not point at a missing report
        logger.info("run_finished run_id=%d status=%s report=%s", self.run_id, status, report_path)
        print(json.dumps(summary(), indent=2, ensure_ascii=False), flush=True)
        return EXIT_CODES[status]


def _refuse_mixed_versions(conn: sqlite3.Connection, version: str, path: Path) -> None:
    """Graph data from other extraction/alias/topic rules must not be silently mixed with this run's (07)."""
    stored = {row[0] for row in conn.execute(
        """SELECT DISTINCT extractor_version FROM crawl_runs r
           WHERE EXISTS (SELECT 1 FROM sources s WHERE s.crawl_run_id = r.id)""")}
    if others := sorted(stored - {version}):
        raise ValueError(f"Database {path} holds items extracted with {', '.join(others)}, but the current extractor "
                         f"is {version}. Changed extraction, alias, or topic rules need an explicit rebuild: ingest "
                         f"into a new database (set MI_DATABASE_PATH) instead of mixing incompatible versions.")


async def ingest(config: PipelineConfig, settings: Settings, config_path: Path, *,
                 crawler_factory: Callable[[PipelineConfig, Settings], Any] | None = None,
                 extractor: Extractor | None = None) -> int:
    """Run one bounded ingest. Exit status: 0 complete, 1 failed, 3 partial (a required source or a seed page
    without a stored item); setup problems, including a database built with other extraction rules, raise
    ValueError."""
    config_path = Path(config_path)
    files = (config_path, config_path.parent / config.aliases_file, config_path.parent / config.topics_file)
    if missing := [str(path) for path in files if not path.is_file()]:
        raise ValueError(f"Required configuration file not found: {', '.join(missing)}")
    config_hash = sha256(b"\0".join(path.read_bytes() for path in files)).hexdigest()
    if extractor is None:
        extractor = _load_extractor(settings.spacy_model, files[1], files[2])
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    try:
        conn = connect(settings.database_path)
    except (SchemaError, sqlite3.Error) as exc:  # e.g. a directory or unreadable file at the database path
        raise ValueError(f"Database setup failed: {exc}") from exc
    try:
        try:
            migrate(conn)
        except SchemaError as exc:
            raise ValueError(f"Database setup failed: {exc}") from exc
        _refuse_mixed_versions(conn, extractor.version, settings.database_path)
        run = _IngestRun(config, settings, conn, extractor, config_path, config_hash, live=crawler_factory is None)
        return await run.execute(crawler_factory or BrowserCrawler)
    finally:
        conn.close()
