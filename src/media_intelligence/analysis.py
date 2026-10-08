"""Read-only graph queries: entity resolution, bounded BFS networks, growth since a cutoff, and degree centrality."""

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from fractions import Fraction

from .models import normalize_name, timestamp

CENTRALITY_DEFINITION = ("score = degree / (N - 1) for N > 1, else 0; degree = distinct neighbours over incoming and "
                         "outgoing edges, ignoring direction and relation type; N counts every node, isolated ones too")


class EntityNotFound(LookupError):
    """No node or alias matches the name, or entity_id is not one of the matching nodes."""


class AmbiguousEntity(LookupError):
    """Several nodes match the name; the caller must choose one by entity_id."""

    def __init__(self, name: str, candidates: list[dict]):
        super().__init__(f"{len(candidates)} entities match {name!r}; pass entity_id to choose one")
        self.candidates = candidates


class EdgeNotFound(LookupError):
    """No edge has the requested ID."""


@contextmanager
def _snapshot(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    # A deferred read transaction pins one WAL snapshot, so every query in a response sees the same commits.
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        if conn.in_transaction:
            conn.execute("ROLLBACK")


def _ids(values: Iterable[int]) -> str:
    # One JSON parameter expanded by json_each() avoids SQLite's bound-variable limit for large ID sets.
    return json.dumps(list(values))


def _page(limit: int, offset: int) -> None:
    if limit < 1 or offset < 0:
        raise ValueError("limit must be positive and offset nonnegative")


def _entity(row: sqlite3.Row, prefix: str = "") -> dict:
    return {"id": str(row[prefix + "id"]), "name": row[prefix + "name"], "type": row[prefix + "entity_type"],
            "mention_count": row[prefix + "mention_count"]}


_EDGE_DETAIL = """SELECT e.id, e.relation_type, e.directed, e.weight, e.first_seen, e.last_seen,
    s.id AS s_id, s.name AS s_name, s.entity_type AS s_entity_type, s.mention_count AS s_mention_count,
    t.id AS t_id, t.name AS t_name, t.entity_type AS t_entity_type, t.mention_count AS t_mention_count
    FROM edges e JOIN nodes s ON s.id = e.source_node_id JOIN nodes t ON t.id = e.target_node_id"""


def _edge_summary(row: sqlite3.Row) -> dict:
    return {"source": _entity(row, "s_"), "target": _entity(row, "t_"), "relation_type": row["relation_type"],
            "directed": bool(row["directed"]), "first_seen": row["first_seen"], "last_seen": row["last_seen"]}


def resolve_entity(conn: sqlite3.Connection, name: str, entity_id: int | None = None) -> sqlite3.Row:
    """Match a name against canonical names and alias keys; entity_id can only choose among those matches."""
    key = normalize_name(name)
    if not key:
        raise EntityNotFound("An entity name is required")
    with _snapshot(conn):
        rows = conn.execute(
            """SELECT id, name, entity_type, mention_count FROM nodes
               WHERE normalized_name = :key OR id IN (SELECT node_id FROM node_aliases WHERE alias_key = :key)
               ORDER BY id""", {"key": key}).fetchall()
    if entity_id is not None:
        rows = [row for row in rows if row["id"] == entity_id]
    if not rows:
        suffix = f" with entity_id {entity_id}" if entity_id is not None else ""
        raise EntityNotFound(f"No entity matches {name!r}{suffix}")
    if len(rows) > 1:
        raise AmbiguousEntity(name, [_entity(row) for row in rows])
    return rows[0]


def _neighbour_ids(conn: sqlite3.Connection, frontier: list[int]) -> set[int]:
    rows = conn.execute(
        """SELECT target_node_id FROM edges WHERE source_node_id IN (SELECT value FROM json_each(:ids))
           UNION SELECT source_node_id FROM edges WHERE target_node_id IN (SELECT value FROM json_each(:ids))""",
        {"ids": _ids(frontier)})
    return {row[0] for row in rows}


def network(conn: sqlite3.Connection, name: str, *, depth: int = 2, entity_id: int | None = None,
            max_nodes: int = 200, max_edges: int = 500) -> dict:
    """Nodes within `depth` hops (either direction) of the resolved entity and the induced edges between them.

    Nodes are kept in (distance, mention_count desc, id) order. Every kept node first gets one edge towards the root
    (see _connected_selection); the remaining edge budget goes in (nearer endpoint distance, farther endpoint
    distance, weight desc, id) order. Anything beyond the limits is dropped and reported as truncated.
    """
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    if max_nodes < 1 or max_edges < 1:
        raise ValueError("max_nodes and max_edges must be positive")
    with _snapshot(conn):
        root = resolve_entity(conn, name, entity_id)
        distance = {root["id"]: 0}
        frontier = [root["id"]]
        for level in range(1, depth + 1):
            # Once a level overflows max_nodes, no farther node can be returned, so stop expanding.
            if not frontier or len(distance) > max_nodes:
                break
            frontier = sorted(_neighbour_ids(conn, frontier) - distance.keys())
            distance.update(dict.fromkeys(frontier, level))
        rows = conn.execute("SELECT id, name, entity_type, mention_count FROM nodes "
                            "WHERE id IN (SELECT value FROM json_each(?))", (_ids(distance),)).fetchall()
        rows.sort(key=lambda row: (distance[row["id"]], -row["mention_count"], row["id"]))
        nodes = rows[:max_nodes]
        kept = _ids(row["id"] for row in nodes)
        edges = conn.execute(
            """SELECT id, source_node_id, target_node_id, relation_type, directed, weight, first_seen, last_seen
               FROM edges WHERE source_node_id IN (SELECT value FROM json_each(:ids))
               AND target_node_id IN (SELECT value FROM json_each(:ids))""", {"ids": kept}).fetchall()

    def edge_order(row: sqlite3.Row) -> tuple:
        ends = sorted((distance[row["source_node_id"]], distance[row["target_node_id"]]))
        return ends[0], ends[1], -row["weight"], row["id"]

    edges.sort(key=edge_order)
    limited = [label for label, hit in (("max_nodes", len(rows) > max_nodes), ("max_edges", len(edges) > max_edges))
               if hit]
    nodes, edges = _connected_selection(nodes, edges, distance, max_edges, edge_order)
    return {
        "root_id": str(root["id"]),
        "depth": depth,
        "nodes": [{"id": str(row["id"]), "name": row["name"], "type": row["entity_type"],
                   "mention_count": row["mention_count"], "distance": distance[row["id"]]} for row in nodes],
        "edges": [{"id": str(row["id"]), "source": str(row["source_node_id"]), "target": str(row["target_node_id"]),
                   "relation_type": row["relation_type"], "directed": bool(row["directed"]), "weight": row["weight"],
                   "first_seen": row["first_seen"], "last_seen": row["last_seen"],
                   "evidence_url": f"/edges/{row['id']}/sources"} for row in edges],
        "meta": {"truncated": bool(limited), "reason": "_and_".join(limited) or None,
                 "max_nodes": max_nodes, "max_edges": max_edges},
    }


def _connected_selection(nodes: list, edges: list, distance: dict[int, int], max_edges: int, edge_order) -> tuple:
    """Edges within max_edges such that every returned node keeps a path to the root.

    Each kept node at distance d > 0 first gets one "parent" edge, its heaviest edge (lowest id on a tie) to a node
    at distance d - 1; nodes are ranked by distance first, so that neighbour is always kept too. The rest of the
    budget is filled in edge_order. If max_edges cannot cover one parent edge per node, the lowest-ranked nodes are
    dropped instead of being returned without an edge.
    """
    parents: dict[int, sqlite3.Row] = {}
    for row in sorted(edges, key=lambda r: (-r["weight"], r["id"])):
        for child, other in ((row["source_node_id"], row["target_node_id"]),
                             (row["target_node_id"], row["source_node_id"])):
            if distance[child] > 0 and distance[other] == distance[child] - 1:
                parents.setdefault(child, row)
    keep = [nodes[0]] + [row for row in nodes[1:] if row["id"] in parents][:max_edges]
    kept = {row["id"] for row in keep}
    chosen = {parents[row["id"]]["id"]: parents[row["id"]] for row in keep[1:]}
    for row in edges:
        if len(chosen) >= max_edges:
            break
        if row["source_node_id"] in kept and row["target_node_id"] in kept:
            chosen.setdefault(row["id"], row)
    return keep, sorted(chosen.values(), key=edge_order)


# t(e, u) = earliest observation of URL u supporting edge e, so extra versions of one URL never add weight.
_GROWTH = """WITH firsts AS (
        SELECT es.edge_id, min(es.observed_at) AS t
        FROM edge_sources es JOIN sources s ON s.id = es.source_id
        GROUP BY es.edge_id, s.source_url)
    SELECT edge_id, sum(t < :since) AS baseline, sum(t >= :since AND t <= :as_of) AS added,
        max(CASE WHEN t >= :since AND t <= :as_of THEN t END) AS latest
    FROM firsts GROUP BY edge_id HAVING added > 0"""


def new_connections(conn: sqlite3.Connection, since: datetime, *, as_of: datetime, min_delta: int, min_ratio: float,
                    limit: int = 20, offset: int = 0) -> dict:
    """Edges first supported in [since, as_of] (new) or whose distinct-URL support grew by the thresholds (grown).

    Baseline B counts URLs first observed before `since`; delta D counts URLs first observed from `since` through
    `as_of` inclusive. Grown means B > 0, D >= min_delta and D / B >= min_ratio, compared exactly as fractions.
    """
    start, end = timestamp(since), timestamp(as_of)
    if start > end:
        raise ValueError("since must not be later than as_of")
    ratio = Fraction(str(min_ratio))
    if min_delta < 1 or ratio <= 0:
        raise ValueError("Growth thresholds must be positive")
    _page(limit, offset)
    with _snapshot(conn):
        found = []
        for row in conn.execute(_GROWTH, {"since": start, "as_of": end}):
            baseline, added = row["baseline"], row["added"]
            if baseline == 0:
                reason = "new"
            elif added >= min_delta and Fraction(added, baseline) >= ratio:
                reason = "grown"
            else:
                continue
            found.append((row["edge_id"], baseline, added, row["latest"], reason))
        # Stable sorts: delta desc, then latest contributing observation desc, then edge id asc.
        found.sort(key=lambda item: item[0])
        found.sort(key=lambda item: item[3], reverse=True)
        found.sort(key=lambda item: item[2], reverse=True)
        page = found[offset:offset + limit]
        details = {row["id"]: row for row in conn.execute(
            _EDGE_DETAIL + " WHERE e.id IN (SELECT value FROM json_each(?))", (_ids(item[0] for item in page),))}
    items = []
    for edge_id, baseline, added, latest, reason in page:
        items.append({
            "edge_id": str(edge_id), **_edge_summary(details[edge_id]), "reason": reason,
            "weight_before": baseline, "weight_now": baseline + added, "weight_delta": added,
            "relative_growth": None if reason == "new" else added / baseline,
            "latest_contribution": latest, "evidence_url": f"/edges/{edge_id}/sources",
        })
    return {"since": start, "as_of": end, "thresholds": {"min_delta": min_delta, "min_ratio": min_ratio},
            "items": items, "total": len(found), "limit": limit, "offset": offset}


_CENTRAL = """WITH incident AS (
        SELECT source_node_id AS node_id, target_node_id AS other_id, relation_type, weight FROM edges
        WHERE source_node_id <> target_node_id
        UNION ALL
        SELECT target_node_id, source_node_id, relation_type, weight FROM edges
        WHERE source_node_id <> target_node_id)
    SELECT n.id, n.name, n.entity_type, n.mention_count, count(DISTINCT i.other_id) AS degree,
        count(DISTINCT i.relation_type) AS relation_type_count, coalesce(sum(i.weight), 0) AS weighted_degree
    FROM nodes n LEFT JOIN incident i ON i.node_id = n.id
    GROUP BY n.id
    ORDER BY degree DESC, relation_type_count DESC, weighted_degree DESC, n.id
    LIMIT ? OFFSET ?"""


def central(conn: sqlite3.Connection, *, limit: int = 20, offset: int = 0) -> dict:
    """Rank nodes by normalized degree; N is constant, so ordering by degree is ordering by score."""
    _page(limit, offset)
    with _snapshot(conn):
        count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        rows = conn.execute(_CENTRAL, (limit, offset)).fetchall()
    items = [{"rank": offset + index, "node_id": str(row["id"]), "name": row["name"], "type": row["entity_type"],
              "score": row["degree"] / (count - 1) if count > 1 else 0.0, "degree": row["degree"],
              "relation_type_count": row["relation_type_count"], "weighted_degree": row["weighted_degree"],
              "mention_count": row["mention_count"]} for index, row in enumerate(rows, start=1)]
    return {"metric": "normalized_degree", "definition": CENTRALITY_DEFINITION, "node_count": count,
            "items": items, "total": count, "limit": limit, "offset": offset}


# An explicit allowlist: source bodies, raw HTML, and crawl metadata never leave the database through the API.
_EVIDENCE_FIELDS = ("source_url", "source_type", "title", "author", "published_at", "scraped_at", "last_scraped_at",
                    "observed_at", "evidence_text", "segment_id", "segment_kind", "segment_author", "segment_url",
                    "segment_published_at", "start_offset", "end_offset", "rule_id", "quality_tier")


def edge_sources(conn: sqlite3.Connection, edge_id: int, *, limit: int = 20, offset: int = 0) -> dict:
    """Every stored source version supporting an edge, oldest observation first; bodies and HTML are not returned."""
    _page(limit, offset)
    with _snapshot(conn):
        edge = conn.execute(_EDGE_DETAIL + " WHERE e.id = ?", (edge_id,)).fetchone()
        if edge is None:
            raise EdgeNotFound(f"No edge has id {edge_id}")
        total = conn.execute("SELECT count(*) FROM edge_sources WHERE edge_id = ?", (edge_id,)).fetchone()[0]
        rows = conn.execute(
            # A thread's comment or reply has its own author, permalink and time, which the evidence must carry:
            # the source row describes only the thread (its submitter and posting time).
            """SELECT es.id, es.source_id, s.source_url, s.source_type, s.title, s.author, s.published_at,
                   s.scraped_at, s.last_scraped_at, es.observed_at, es.evidence_text, es.segment_id,
                   json_extract(seg.value, '$.kind') AS segment_kind,
                   json_extract(seg.value, '$.author') AS segment_author,
                   json_extract(seg.value, '$.url') AS segment_url,
                   json_extract(seg.value, '$.published_at') AS segment_published_at,
                   es.start_offset, es.end_offset, es.rule_id, es.quality_tier
               FROM edge_sources es JOIN sources s ON s.id = es.source_id
               LEFT JOIN json_each(s.segments_json) seg ON json_extract(seg.value, '$.id') = es.segment_id
               WHERE es.edge_id = ? ORDER BY es.observed_at, es.id LIMIT ? OFFSET ?""",
            (edge_id, limit, offset)).fetchall()
    items = [{"evidence_id": str(row["id"]), "source_id": str(row["source_id"]),
              **{key: row[key] for key in _EVIDENCE_FIELDS}} for row in rows]
    return {"edge": {"id": str(edge["id"]), **_edge_summary(edge), "weight": edge["weight"]},
            "items": items, "total": total, "limit": limit, "offset": offset}


def graph_counts(conn: sqlite3.Connection) -> dict[str, int]:
    with _snapshot(conn):
        row = conn.execute("""SELECT (SELECT count(*) FROM nodes) AS nodes, (SELECT count(*) FROM edges) AS edges,
            (SELECT count(*) FROM sources) AS sources, (SELECT count(*) FROM edge_sources) AS evidence""").fetchone()
    return dict(row)
