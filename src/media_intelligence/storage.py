"""Write path: one transaction per content item stores its source version, nodes, mentions, aliases, edges, evidence."""

import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from .db import integrity_report, transaction
from .models import ContentItem, Extraction, Relation, node_key, normalize_name, timestamp, utcnow

logger = logging.getLogger(__name__)

RUN_STATUSES = frozenset({"complete", "partial", "failed"})
# Priority order when one surface form is seen with several resolutions; anything else is a plain surface form.
# "alias" is a reviewed alias from aliases.toml; origin 'reviewed' marks rows taken from that table itself.
ALIAS_KINDS = ("handle", "abbreviation", "alias", "surname", "surface")


@dataclass
class StoreResult:
    source_id: int
    new_version: bool
    nodes_created: int = 0
    edges_created: int = 0
    evidence_added: int = 0
    mentions_added: int = 0
    relations_skipped: int = 0  # an endpoint has no mention in this extraction
    self_edges_rejected: int = 0
    invalid_evidence: int = 0  # unknown segment, offsets out of range, or text not equal to the segment slice


@dataclass
class _Node:
    entity_type: str
    name: str
    spans: set[tuple[str, int, int]] = field(default_factory=set)
    surfaces: set[str] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def start_run(conn: sqlite3.Connection, *, config_hash: str, extractor_version: str, started_at: datetime) -> int:
    # lastrowid instead of RETURNING: an unexhausted RETURNING cursor would hold the autocommit write open.
    return conn.execute("INSERT INTO crawl_runs (started_at, config_hash, extractor_version) VALUES (?, ?, ?)",
                        (timestamp(started_at), config_hash, extractor_version)).lastrowid


def finish_run(conn: sqlite3.Connection, run_id: int, *, status: str, summary: dict, finished_at: datetime) -> None:
    if status not in RUN_STATUSES:
        raise ValueError(f"Run status must be one of {sorted(RUN_STATUSES)}, not {status!r}")
    cursor = conn.execute("UPDATE crawl_runs SET status = ?, summary_json = ?, finished_at = ? WHERE id = ?",
                          (status, _json(summary), timestamp(finished_at), run_id))
    if cursor.rowcount != 1:
        raise ValueError(f"Unknown crawl run {run_id}")


def source_version_id(conn: sqlite3.Connection, item: ContentItem) -> int | None:
    """Id of the stored version with this URL and content hash; lets a caller skip extraction for unchanged items."""
    row = conn.execute("SELECT id FROM sources WHERE source_url = ? AND content_hash = ?",
                       (item.source_url, item.content_hash)).fetchone()
    return row[0] if row else None


def _alias_kind(resolution: str) -> str:
    resolution = resolution.lower()
    return next((kind for kind in ALIAS_KINDS[:-1] if kind in resolution), "surface")


def _collect_nodes(extraction: Extraction) -> dict[str, _Node]:
    nodes: dict[str, _Node] = {}
    for mention in extraction.mentions:
        key = node_key(mention.entity_type, mention.canonical_name)
        if key != mention.key or not key.partition(":")[2]:
            raise ValueError(f"Mention key {mention.key!r} does not match its type and canonical name ({key!r})")
        # The first mention in extraction order supplies the display name of a node created by this item.
        node = nodes.setdefault(key, _Node(mention.entity_type, " ".join(mention.canonical_name.split())))
        node.spans.add((mention.segment_id, mention.start, mention.end))
        if mention.name.strip():
            node.surfaces.add(mention.name)
        if alias := normalize_name(mention.name):
            kind = _alias_kind(mention.resolution)
            node.aliases[alias] = min(kind, node.aliases.get(alias, kind), key=ALIAS_KINDS.index)
    return nodes


def _valid_evidence(relation: Relation, segments: dict[str, str]) -> bool:
    text = segments.get(relation.segment_id)
    return (text is not None and 0 <= relation.start < relation.end <= len(text)
            and text[relation.start:relation.end] == relation.evidence_text and bool(relation.evidence_text.strip())
            and bool(relation.relation_type.strip()) and bool(relation.rule_id.strip()))


def _collect_relations(item: ContentItem, extraction: Extraction, keys) -> tuple[list[Relation], Counter]:
    # Offsets are relative to the exact stored segment text; an item without segments has one implicit body segment.
    segments = {segment.id: segment.text for segment in item.segments} or {"body": item.body}
    valid, rejected = [], Counter()
    for relation in extraction.relations:
        if relation.source_key not in keys or relation.target_key not in keys:
            rejected["relations_skipped"] += 1
        elif relation.source_key == relation.target_key:
            rejected["self_edges_rejected"] += 1
        elif not _valid_evidence(relation, segments):
            rejected["invalid_evidence"] += 1
        else:
            valid.append(relation)
    if rejected:
        logger.warning("Rejected relations for %s: %s", item.source_url, dict(rejected))
    return valid, rejected


def _refresh_version(conn: sqlite3.Connection, source_id: int, seen: str) -> None:
    conn.execute("UPDATE sources SET last_scraped_at = max(last_scraped_at, ?) WHERE id = ?", (seen, source_id))
    conn.execute("""UPDATE edges SET last_seen = max(last_seen, ?)
                    WHERE id IN (SELECT edge_id FROM edge_sources WHERE source_id = ?)""", (seen, source_id))


def _insert_source(conn: sqlite3.Connection, item: ContentItem, content_hash: str, run_id: int, seen: str) -> int:
    return conn.execute(
        """INSERT INTO sources (source_url, requested_url, source_type, scraped_at, last_scraped_at, title, body,
           author, published_at, content_hash, segments_json, metadata_json, crawl_run_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (item.source_url, item.requested_url or item.source_url, item.source_type, seen, seen, item.title, item.body,
         item.author, timestamp(item.published_at) if item.published_at else None, content_hash,
         json.dumps([segment.model_dump(mode="json") for segment in item.segments], ensure_ascii=False),
         _json(item.metadata), run_id)).lastrowid


def _upsert_node(conn: sqlite3.Connection, key: str, node: _Node, seen: str) -> tuple[int, bool]:
    # An existing node keeps its first display name so IDs and labels stay stable across runs.
    row = conn.execute("SELECT id FROM nodes WHERE key = ?", (key,)).fetchone()
    if row:
        return row[0], False
    return conn.execute(
        "INSERT INTO nodes (key, name, normalized_name, entity_type, first_seen) VALUES (?, ?, ?, ?, ?)",
        (key, node.name, key.partition(":")[2], node.entity_type, seen)).lastrowid, True


def _upsert_edge(conn: sqlite3.Connection, source: int, target: int, relation: Relation, seen: str) -> tuple[int, bool]:
    row = conn.execute("""SELECT id, directed FROM edges
                          WHERE source_node_id = ? AND target_node_id = ? AND relation_type = ?""",
                       (source, target, relation.relation_type)).fetchone()
    if row is None:
        return conn.execute(
            """INSERT INTO edges (source_node_id, target_node_id, relation_type, directed, weight, first_seen,
               last_seen) VALUES (?, ?, ?, ?, 1, ?, ?)""",
            (source, target, relation.relation_type, int(relation.directed), seen, seen)).lastrowid, True
    if bool(row["directed"]) != relation.directed:
        raise ValueError(f"Relation {relation.relation_type!r} is stored with directed={bool(row['directed'])} "
                         f"but was extracted with directed={relation.directed}")
    return row["id"], False


def _update_aggregates(conn: sqlite3.Connection, node_ids, edge_ids, seen: str) -> None:
    conn.executemany(
        """UPDATE nodes SET
               mention_count = (SELECT count(DISTINCT s.source_url) FROM node_mentions nm
                                JOIN sources s ON s.id = nm.source_id WHERE nm.node_id = nodes.id),
               first_seen = (SELECT min(first_observed_at) FROM node_mentions WHERE node_id = nodes.id)
           WHERE id = ?""", [(node_id,) for node_id in sorted(node_ids)])
    conn.executemany(
        """UPDATE edges SET
               weight = (SELECT count(DISTINCT s.source_url) FROM edge_sources es
                         JOIN sources s ON s.id = es.source_id WHERE es.edge_id = edges.id),
               first_seen = (SELECT min(observed_at) FROM edge_sources WHERE edge_id = edges.id),
               last_seen = max(last_seen, ?)
           WHERE id = ?""", [(seen, edge_id) for edge_id in sorted(edge_ids)])


def store_item(conn: sqlite3.Connection, item: ContentItem, extraction: Extraction, *, run_id: int,
               observed_at: datetime | None = None,
               reviewed_aliases: Mapping[str, Iterable[tuple[str, str]]] | None = None) -> StoreResult:
    """Store one item atomically. A known (URL, content hash) version only refreshes recency and adds no support.

    Without observed_at, the source keeps item.scraped_at (fetch time) while mentions and evidence are stamped with
    the clock read after the write lock is taken (never before the fetch). A reader polling with since=<its last
    as_of> then cannot miss an observation committed after it read; only the commit itself remains as a window.
    An explicit observed_at stamps everything with that one time (tests, replays). reviewed_aliases maps node keys
    to configured (alias key, kind) pairs, stored with origin 'reviewed' for every node this item touches.
    """
    content_hash = item.content_hash
    nodes = _collect_nodes(extraction)
    relations, rejected = _collect_relations(item, extraction, nodes.keys())
    with transaction(conn):
        if observed_at is None:
            scraped, seen = timestamp(item.scraped_at), timestamp(max(utcnow(), item.scraped_at))
        else:
            scraped = seen = timestamp(observed_at)
        row = conn.execute("SELECT id FROM sources WHERE source_url = ? AND content_hash = ?",
                           (item.source_url, content_hash)).fetchone()
        if row:
            _refresh_version(conn, row[0], scraped)
            return StoreResult(row[0], new_version=False)
        result = StoreResult(_insert_source(conn, item, content_hash, run_id, scraped), new_version=True, **rejected)
        node_ids: dict[str, int] = {}
        for key, node in nodes.items():
            node_ids[key], created = _upsert_node(conn, key, node, seen)
            result.nodes_created += created
            conn.execute("""INSERT INTO node_mentions (node_id, source_id, first_observed_at, surface_forms_json,
                            occurrence_count) VALUES (?, ?, ?, ?, ?)""",
                         (node_ids[key], result.source_id, seen, json.dumps(sorted(node.surfaces), ensure_ascii=False),
                          len(node.spans)))
            result.mentions_added += 1
            conn.executemany("""INSERT INTO node_aliases (alias_key, node_id, alias_kind, origin)
                                VALUES (?, ?, ?, 'reviewed') ON CONFLICT (alias_key, node_id) DO NOTHING""",
                             [(alias, node_ids[key], kind) for alias, kind in (reviewed_aliases or {}).get(key, ())])
            conn.executemany("""INSERT INTO node_aliases (alias_key, node_id, alias_kind, origin)
                                VALUES (?, ?, ?, 'observed') ON CONFLICT (alias_key, node_id) DO NOTHING""",
                             [(alias, node_ids[key], kind) for alias, kind in sorted(node.aliases.items())])
        edge_ids: set[int] = set()
        for relation in relations:
            source, target = node_ids[relation.source_key], node_ids[relation.target_key]
            if not relation.directed and source > target:
                source, target = target, source
            edge_id, created = _upsert_edge(conn, source, target, relation, seen)
            result.edges_created += created
            edge_ids.add(edge_id)
            # One representative excerpt per edge and version: the first relation in extraction order wins.
            result.evidence_added += conn.execute(
                """INSERT INTO edge_sources (edge_id, source_id, observed_at, evidence_text, segment_id, start_offset,
                   end_offset, rule_id, quality_tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (edge_id, source_id) DO NOTHING""",
                (edge_id, result.source_id, seen, relation.evidence_text, relation.segment_id, relation.start,
                 relation.end, relation.rule_id, relation.quality_tier)).rowcount
        _update_aggregates(conn, node_ids.values(), edge_ids, seen)
    return result


_STORAGE_CHECKS = {
    "evidence_observed_before_source_scraped": """SELECT count(*) FROM edge_sources es
        JOIN sources s ON s.id = es.source_id WHERE es.observed_at < s.scraped_at""",
    "mentions_observed_before_source_scraped": """SELECT count(*) FROM node_mentions nm
        JOIN sources s ON s.id = nm.source_id WHERE nm.first_observed_at < s.scraped_at""",
    "edge_last_seen_stale": """SELECT count(*) FROM edges e WHERE e.last_seen < (
        SELECT max(s.last_scraped_at) FROM edge_sources es JOIN sources s ON s.id = es.source_id
        WHERE es.edge_id = e.id)""",
}


def verify_integrity(conn: sqlite3.Connection) -> list[str]:
    """Foreign-key and evidence-derived count/time checks as 'check: count' messages; an empty list means OK."""
    report = integrity_report(conn)
    report.update({name: conn.execute(sql).fetchone()[0] for name, sql in _STORAGE_CHECKS.items()})
    return [f"{name}: {count}" for name, count in report.items() if count]


def graph_counts(conn: sqlite3.Connection) -> dict:
    """Run-report totals. An edge is counted under every quality tier that has evidence for it."""
    def scalar(sql: str) -> int:
        return conn.execute(sql).fetchone()[0]

    def grouped(sql: str) -> dict[str, int]:
        return {name: count for name, count in conn.execute(sql).fetchall()}

    def nested(sql: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for outer, inner, count in conn.execute(sql).fetchall():
            result.setdefault(outer, {})[inner] = count
        return result

    return {
        "sources": scalar("SELECT count(*) FROM sources"),
        "source_urls": scalar("SELECT count(DISTINCT source_url) FROM sources"),
        "nodes": scalar("SELECT count(*) FROM nodes"),
        "nodes_by_type": grouped("SELECT entity_type, count(*) FROM nodes GROUP BY 1 ORDER BY 1"),
        "aliases": scalar("SELECT count(*) FROM node_aliases"),
        "edges": scalar("SELECT count(*) FROM edges"),
        "edges_by_relation_type": grouped("SELECT relation_type, count(*) FROM edges GROUP BY 1 ORDER BY 1"),
        "edges_by_quality_tier": grouped(
            "SELECT quality_tier, count(DISTINCT edge_id) FROM edge_sources GROUP BY 1 ORDER BY 1"),
        "evidence": scalar("SELECT count(*) FROM edge_sources"),
        "evidence_by_quality_tier": grouped("SELECT quality_tier, count(*) FROM edge_sources GROUP BY 1 ORDER BY 1"),
        # Whether every source type contributes typed edges or only co-occurrence (planning doc 15).
        "evidence_by_source_type_and_tier": nested(
            """SELECT s.source_type, es.quality_tier, count(*) FROM edge_sources es JOIN sources s ON s.id = es.source_id
               GROUP BY 1, 2 ORDER BY 1, 2"""),
        "semantic_evidence_by_source_type_and_relation": nested(
            """SELECT s.source_type, e.relation_type, count(*) FROM edge_sources es JOIN sources s ON s.id = es.source_id
               JOIN edges e ON e.id = es.edge_id WHERE es.quality_tier = 'semantic_rule' GROUP BY 1, 2 ORDER BY 1, 2"""),
    }
