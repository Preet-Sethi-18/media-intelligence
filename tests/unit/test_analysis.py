"""Graph query tests on temporary SQLite files filled with small, hand-made synthetic rows (not crawled data)."""

import json
from datetime import UTC, datetime, timedelta, timezone
from itertools import count

import pytest

from media_intelligence import analysis, db
from media_intelligence.models import node_key, normalize_name, timestamp

T0 = datetime(2026, 10, 1, 12, tzinfo=UTC)
SINCE = datetime(2026, 10, 7, 12, tzinfo=UTC)
AS_OF = datetime(2026, 10, 8, 12, tzinfo=UTC)
US = timedelta(microseconds=1)


class Graph:
    """Builds a synthetic graph whose edge weights and times are derived from the evidence rows it inserts."""

    def __init__(self, conn):
        self.conn = conn
        self.hashes = count()
        self.run = conn.execute("INSERT INTO crawl_runs (started_at, config_hash, extractor_version) "
                                "VALUES (?, 'cfg', 'test')", (timestamp(T0),)).lastrowid

    def node(self, name, entity_type="PERSON", mentions=1, aliases=()):
        node_id = self.conn.execute(
            """INSERT INTO nodes (key, name, normalized_name, entity_type, first_seen, mention_count)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node_key(entity_type, name), name, normalize_name(name), entity_type, timestamp(T0), mentions)).lastrowid
        for alias in aliases:
            self.conn.execute("INSERT INTO node_aliases (alias_key, node_id, alias_kind, origin) "
                              "VALUES (?, ?, 'handle', 'test')", (normalize_name(alias), node_id))
        return node_id

    def edge(self, source, target, relation="met_with", directed=False, observations=None):
        """observations: (url, observed_at) pairs; repeating a URL stores another version of that URL."""
        observations = observations or [(f"https://example.com/{source}-{target}-{relation}", T0)]
        if not directed:
            source, target = sorted((source, target))
        times = [observed for _, observed in observations]
        edge_id = self.conn.execute(
            """INSERT INTO edges (source_node_id, target_node_id, relation_type, directed, weight, first_seen,
               last_seen) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source, target, relation, int(directed), len({url for url, _ in observations}),
             timestamp(min(times)), timestamp(max(times)))).lastrowid
        for url, observed in observations:
            source_id = self.source(url, observed)
            self.conn.execute(
                """INSERT INTO edge_sources (edge_id, source_id, observed_at, evidence_text, segment_id,
                   start_offset, end_offset, rule_id, quality_tier)
                   VALUES (?, ?, ?, 'Synthetic evidence sentence.', 'body', 0, 28, 'test_rule', 'semantic_rule')""",
                (edge_id, source_id, timestamp(observed)))
        return edge_id

    def source(self, url, observed):
        return self.conn.execute(
            """INSERT INTO sources (source_url, requested_url, source_type, scraped_at, last_scraped_at, title, body,
               content_hash, crawl_run_id) VALUES (?, ?, 'news', ?, ?, 'Synthetic title', 'Synthetic body.', ?, ?)""",
            (url, url, timestamp(observed), timestamp(observed), f"hash-{next(self.hashes)}", self.run)).lastrowid


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "graph.sqlite3")
    db.migrate(connection)
    yield connection
    connection.close()


def build(conn, fill):
    with db.transaction(conn):
        return fill(Graph(conn))


@pytest.fixture
def sample(conn):
    """Synthetic graph: A-B, C->A, B->D, C-D (cycle A-B-D-C), D->E (3 hops from A), A->B criticized, Z isolated."""
    def fill(g):
        ids = {name: g.node(name, mentions=m) for name, m in
               [("Alice Example", 5), ("Bravo Org", 1), ("Charlie Example", 3), ("Delta Org", 2), ("Echo", 1),
                ("Zulu Isolated", 1)]}
        a, b, c, d, e = (ids[n] for n in ["Alice Example", "Bravo Org", "Charlie Example", "Delta Org", "Echo"])
        edges = {"ab": g.edge(a, b), "ca": g.edge(c, a, "criticized", True), "bd": g.edge(b, d, "affiliated_with", True),
                 "cd": g.edge(c, d), "de": g.edge(d, e, "criticized", True), "ab2": g.edge(a, b, "criticized", True)}
        g.conn.execute("INSERT INTO node_aliases VALUES ('@alice', ?, 'handle', 'test')", (a,))
        return ids, edges
    return build(conn, fill)


def ids_of(items):
    return [item["id"] for item in items]


def test_depth_one_returns_root_and_direct_neighbours_in_either_direction(conn, sample):
    ids, edges = sample
    result = analysis.network(conn, "Alice Example", depth=1)
    assert result["root_id"] == str(ids["Alice Example"]) and result["depth"] == 1
    assert {n["name"]: n["distance"] for n in result["nodes"]} == {
        "Alice Example": 0, "Bravo Org": 1, "Charlie Example": 1}
    assert sorted(ids_of(result["edges"])) == sorted(str(edges[k]) for k in ("ab", "ca", "ab2"))
    incoming = next(e for e in result["edges"] if e["id"] == str(edges["ca"]))
    assert incoming == {"id": str(edges["ca"]), "source": str(ids["Charlie Example"]),
                        "target": str(ids["Alice Example"]), "relation_type": "criticized", "directed": True,
                        "weight": 1, "first_seen": timestamp(T0), "last_seen": timestamp(T0),
                        "evidence_url": f"/edges/{edges['ca']}/sources"}
    assert result["meta"] == {"truncated": False, "reason": None, "max_nodes": 200, "max_edges": 500}
    assert not conn.in_transaction


def test_depth_two_adds_second_hop_through_cycle_and_only_induced_edges(conn, sample):
    _, edges = sample
    result = analysis.network(conn, "alice   EXAMPLE", depth=2)
    assert {n["name"]: n["distance"] for n in result["nodes"]} == {
        "Alice Example": 0, "Bravo Org": 1, "Charlie Example": 1, "Delta Org": 2}
    assert sorted(ids_of(result["edges"])) == sorted(str(edges[k]) for k in ("ab", "ca", "bd", "cd", "ab2"))
    node_ids = set(ids_of(result["nodes"]))
    assert len(node_ids) == len(result["nodes"]) and len(set(ids_of(result["edges"]))) == len(result["edges"])
    assert all(e["source"] in node_ids and e["target"] in node_ids for e in result["edges"])
    # Deterministic order: distance, then mention_count desc, then id; edges nearest the root first.
    assert [n["name"] for n in result["nodes"]] == ["Alice Example", "Charlie Example", "Bravo Org", "Delta Org"]
    assert ids_of(result["edges"])[:3] == [str(edges[k]) for k in ("ab", "ca", "ab2")]
    undirected = next(e for e in result["edges"] if e["id"] == str(edges["ab"]))
    assert undirected["directed"] is False and undirected["relation_type"] == "met_with"


def test_isolated_node_returns_itself_and_no_edges(conn, sample):
    ids, _ = sample
    result = analysis.network(conn, "Zulu Isolated", depth=2)
    assert result["nodes"] == [{"id": str(ids["Zulu Isolated"]), "name": "Zulu Isolated", "type": "PERSON",
                                "mention_count": 1, "distance": 0}]
    assert result["edges"] == [] and result["meta"]["truncated"] is False


def test_alias_and_handle_resolve_to_the_node(conn, sample):
    ids, _ = sample
    assert analysis.resolve_entity(conn, "@Alice")["id"] == ids["Alice Example"]
    assert analysis.network(conn, "@alice", depth=1)["root_id"] == str(ids["Alice Example"])


def test_node_limit_truncates_deterministically_without_dangling_edges(conn, sample):
    _, edges = sample
    result = analysis.network(conn, "Alice Example", depth=2, max_nodes=2)
    assert [n["name"] for n in result["nodes"]] == ["Alice Example", "Charlie Example"]
    assert ids_of(result["edges"]) == [str(edges["ca"])]
    assert result["meta"] == {"truncated": True, "reason": "max_nodes", "max_nodes": 2, "max_edges": 500}
    exact = analysis.network(conn, "Alice Example", depth=1, max_nodes=3)
    assert exact["meta"]["truncated"] is False
    second_hop_cut = analysis.network(conn, "Alice Example", depth=2, max_nodes=3)
    assert len(second_hop_cut["nodes"]) == 3 and second_hop_cut["meta"]["reason"] == "max_nodes"


def test_edge_limit_truncates_and_reports_both_limits(conn, sample):
    _, edges = sample
    result = analysis.network(conn, "Alice Example", depth=2, max_edges=2)
    assert ids_of(result["edges"]) == [str(edges["ab"]), str(edges["ca"])]
    assert result["meta"]["truncated"] is True and result["meta"]["reason"] == "max_edges"
    both = analysis.network(conn, "Alice Example", depth=2, max_nodes=3, max_edges=1)
    assert both["meta"]["reason"] == "max_nodes_and_max_edges"


def test_ambiguous_name_needs_entity_id_and_unknown_names_fail(conn):
    person, place = build(conn, lambda g: (g.node("Jordan", "PERSON"), g.node("Jordan", "LOCATION")))
    with pytest.raises(analysis.AmbiguousEntity) as caught:
        analysis.resolve_entity(conn, "jordan")
    assert caught.value.candidates == [
        {"id": str(person), "name": "Jordan", "type": "PERSON", "mention_count": 1},
        {"id": str(place), "name": "Jordan", "type": "LOCATION", "mention_count": 1}]
    assert analysis.resolve_entity(conn, "Jordan", entity_id=place)["entity_type"] == "LOCATION"
    assert analysis.network(conn, "Jordan", entity_id=person)["root_id"] == str(person)
    other = build(conn, lambda g: g.node("Someone Else"))
    for name, entity_id in [("Jordan", other), ("Jordan", 999), ("Nobody", None), ("   ", None)]:
        with pytest.raises(analysis.EntityNotFound):
            analysis.resolve_entity(conn, name, entity_id)


def test_alias_shared_by_two_nodes_is_ambiguous(conn):
    def fill(g):
        return g.node("Elon Musk", aliases=["Musk"]), g.node("Kimbal Musk", aliases=["Musk"])
    first, second = build(conn, fill)
    with pytest.raises(analysis.AmbiguousEntity) as caught:
        analysis.network(conn, "musk")
    assert [c["id"] for c in caught.value.candidates] == [str(first), str(second)]


def test_network_rejects_unsupported_depth(conn, sample):
    with pytest.raises(ValueError):
        analysis.network(conn, "Alice Example", depth=3)


# (baseline URLs, added URLs, expected reason) from the illustrative table in 16_graph_analysis.md.
GROWTH_TABLE = [(0, 1, "new"), (1, 1, None), (2, 3, "grown"), (6, 3, "grown"), (10, 3, None), (10, 5, "grown"),
                (3, 0, None)]


def growth_edge(g, label, before, added, latest=SINCE + timedelta(hours=1)):
    a, b = g.node(f"{label} Source"), g.node(f"{label} Target")
    old = [(f"https://example.com/{label}/old/{i}", SINCE - timedelta(days=1, minutes=i)) for i in range(before)]
    new = [(f"https://example.com/{label}/new/{i}", latest - timedelta(minutes=i)) for i in range(added)]
    return g.edge(a, b, observations=old + new)


def growth(conn, since=SINCE, as_of=AS_OF, **kwargs):
    return analysis.new_connections(conn, since, as_of=as_of, **{"min_delta": 3, "min_ratio": 0.5, **kwargs})


def test_growth_table_from_design(conn):
    edges = build(conn, lambda g: [growth_edge(g, f"row{i}", b, d, SINCE + timedelta(hours=i))
                                   for i, (b, d, _) in enumerate(GROWTH_TABLE)])
    result = growth(conn)
    found = {item["edge_id"]: item for item in result["items"]}
    expected = {str(edge): row for edge, row in zip(edges, GROWTH_TABLE) if row[2]}
    assert found.keys() == expected.keys() and result["total"] == 4
    for edge_id, (before, added, reason) in expected.items():
        item = found[edge_id]
        assert (item["reason"], item["weight_before"], item["weight_delta"], item["weight_now"]) == (
            reason, before, added, before + added)
        assert item["relative_growth"] == (None if reason == "new" else added / before)
    # Delta desc, then latest contributing observation desc (row 3 is later than row 2), then id.
    assert [item["edge_id"] for item in result["items"]] == [str(edges[i]) for i in (5, 3, 2, 0)]
    assert found[str(edges[3])]["relative_growth"] == 0.5
    assert result["thresholds"] == {"min_delta": 3, "min_ratio": 0.5}
    assert (result["since"], result["as_of"]) == (timestamp(SINCE), timestamp(AS_OF))


def test_growth_item_shape(conn):
    edge = build(conn, lambda g: growth_edge(g, "shape", 0, 1))
    item = growth(conn)["items"][0]
    assert item["source"]["name"] == "shape Source" and item["target"]["type"] == "PERSON"
    assert item["evidence_url"] == f"/edges/{edge}/sources" and item["directed"] is False
    assert item["latest_contribution"] == timestamp(SINCE + timedelta(hours=1))
    assert set(item) == {"edge_id", "source", "target", "relation_type", "directed", "first_seen", "last_seen",
                         "reason", "weight_before", "weight_now", "weight_delta", "relative_growth",
                         "latest_contribution", "evidence_url"}


def test_observation_exactly_at_since_is_new_and_one_microsecond_earlier_is_baseline(conn):
    def fill(g):
        at = g.edge(g.node("At A"), g.node("At B"), observations=[("https://example.com/at", SINCE)])
        before = g.edge(g.node("Before A"), g.node("Before B"), observations=[("https://example.com/b", SINCE - US)])
        return at, before
    at, before = build(conn, fill)
    result = growth(conn)
    assert [(item["edge_id"], item["reason"]) for item in result["items"]] == [(str(at), "new")]
    offset = SINCE.astimezone(timezone(timedelta(hours=5, minutes=30)))
    assert [item["edge_id"] for item in growth(conn, since=offset)["items"]] == [str(at)]
    assert growth(conn, since=SINCE + US)["items"] == []
    assert [item["edge_id"] for item in growth(conn, since=SINCE - US)["items"]] == [str(at), str(before)]


def test_as_of_is_inclusive_and_later_observations_are_ignored(conn):
    def fill(g):
        inclusive = g.edge(g.node("End A"), g.node("End B"), observations=[("https://example.com/end", AS_OF)])
        later = g.edge(g.node("Late A"), g.node("Late B"), observations=[("https://example.com/late", AS_OF + US)])
        return inclusive, later
    inclusive, _ = build(conn, fill)
    assert [item["edge_id"] for item in growth(conn)["items"]] == [str(inclusive)]


def test_multiple_versions_of_one_url_count_once_at_their_earliest_observation(conn):
    def fill(g):
        url = "https://example.com/edited"
        old = g.edge(g.node("Old A"), g.node("Old B"),
                     observations=[(url, SINCE - timedelta(hours=1)), (url, SINCE + timedelta(hours=1))])
        fresh = [(f"https://example.com/fresh/{i}", SINCE + timedelta(minutes=i)) for i in range(3)]
        new = g.edge(g.node("New A"), g.node("New B"), observations=fresh + [fresh[0][:1] + (AS_OF,)])
        grown = [(f"https://example.com/grown/{i}", SINCE + timedelta(minutes=i)) for i in range(3)]
        base = [(f"https://example.com/base/{i}", T0) for i in range(2)]
        return old, new, g.edge(g.node("Grow A"), g.node("Grow B"), observations=base + grown + grown)
    old, new, grown = build(conn, fill)
    found = {item["edge_id"]: item for item in growth(conn)["items"]}
    assert str(old) not in found
    assert (found[str(new)]["weight_now"], found[str(new)]["reason"]) == (3, "new")
    assert (found[str(grown)]["weight_before"], found[str(grown)]["weight_delta"]) == (2, 3)


def test_growth_ratio_boundary_uses_exact_arithmetic(conn):
    edge = build(conn, lambda g: growth_edge(g, "ratio", 10, 3))
    assert [item["edge_id"] for item in growth(conn, min_ratio=0.3)["items"]] == [str(edge)]
    assert growth(conn, min_ratio=0.30000000000000004)["items"] == []
    assert growth(conn, min_delta=4, min_ratio=0.3)["items"] == []


def test_growth_pagination_and_validation(conn):
    build(conn, lambda g: [growth_edge(g, f"page{i}", 0, 1, SINCE + timedelta(minutes=i)) for i in range(5)])
    first = growth(conn, limit=2)
    second = growth(conn, limit=2, offset=2)
    assert first["total"] == second["total"] == 5 and len(first["items"]) == len(second["items"]) == 2
    assert not {i["edge_id"] for i in first["items"]} & {i["edge_id"] for i in second["items"]}
    assert growth(conn, offset=10)["items"] == []
    with pytest.raises(ValueError):
        growth(conn, since=SINCE.replace(tzinfo=None))
    with pytest.raises(ValueError):
        growth(conn, since=AS_OF + US)
    with pytest.raises(ValueError):
        growth(conn, limit=0)


def test_centrality_star_with_isolated_node_and_repeated_relations(conn):
    def fill(g):
        hub, *leaves, lonely = (g.node(n) for n in ["Hub", "Leaf One", "Leaf Two", "Leaf Three", "Lonely"])
        for leaf in leaves:
            g.edge(hub, leaf)
        g.edge(hub, leaves[0], "criticized", True)
        g.edge(leaves[0], hub, "criticized", True)
        return hub, leaves, lonely
    hub, leaves, lonely = build(conn, fill)
    result = analysis.central(conn)
    assert (result["metric"], result["node_count"], result["total"]) == ("normalized_degree", 5, 5)
    top = result["items"][0]
    assert top == {"rank": 1, "node_id": str(hub), "name": "Hub", "type": "PERSON", "score": 0.75, "degree": 3,
                   "relation_type_count": 2, "weighted_degree": 5, "mention_count": 1}
    leaf_one = result["items"][1]
    assert (leaf_one["node_id"], leaf_one["degree"], leaf_one["score"]) == (str(leaves[0]), 1, 0.25)
    assert result["items"][-1]["node_id"] == str(lonely) and result["items"][-1]["score"] == 0.0


def test_centrality_tie_breaks_and_pagination(conn):
    def fill(g):
        a, b, c, d, e, f = (g.node(n) for n in "ABCDEF")
        g.edge(a, e)
        g.edge(a, f, "criticized", True)
        g.edge(b, e)
        g.edge(b, f, observations=[(f"https://example.com/bf/{i}", T0) for i in range(3)])
        for node in (c, d):
            g.edge(node, e)
            g.edge(node, f)
        return a, b, c, d, e, f
    a, b, c, d, e, f = build(conn, fill)
    items = analysis.central(conn)["items"]
    assert [i["node_id"] for i in items] == [str(x) for x in (f, e, a, b, c, d)]
    assert [(i["degree"], i["relation_type_count"], i["weighted_degree"]) for i in items] == [
        (4, 2, 6), (4, 1, 4), (2, 2, 2), (2, 1, 4), (2, 1, 2), (2, 1, 2)]
    assert [i["score"] for i in items] == [0.8, 0.8, 0.4, 0.4, 0.4, 0.4]
    page = analysis.central(conn, limit=2, offset=2)
    assert [(i["rank"], i["node_id"]) for i in page["items"]] == [(3, str(a)), (4, str(b))]


def test_centrality_chain_single_node_and_empty_graph(conn):
    assert analysis.central(conn)["items"] == [] and analysis.central(conn)["node_count"] == 0
    only = build(conn, lambda g: g.node("Only"))
    assert analysis.central(conn)["items"][0]["score"] == 0.0
    def fill(g):
        middle, end = g.node("Middle"), g.node("End")
        g.edge(only, middle)
        g.edge(middle, end, "criticized", True)
        return middle, end
    middle, end = build(conn, fill)
    scores = {i["node_id"]: i["score"] for i in analysis.central(conn)["items"]}
    assert scores == {str(middle): 1.0, str(only): 0.5, str(end): 0.5}


def test_edge_sources_lists_every_version_with_total(conn):
    url = "https://example.com/versioned"
    edge = build(conn, lambda g: g.edge(g.node("Ev A"), g.node("Ev B"), "criticized", True, observations=[
        (url, T0 + timedelta(hours=2)), (url, T0), ("https://example.com/other", T0 + timedelta(hours=1))]))
    result = analysis.edge_sources(conn, edge)
    assert result["total"] == 3 and result["edge"]["weight"] == 2 and result["edge"]["id"] == str(edge)
    assert [i["observed_at"] for i in result["items"]] == [timestamp(T0 + timedelta(hours=h)) for h in (0, 1, 2)]
    item = result["items"][0]
    assert item["source_url"] == url and item["evidence_text"] == "Synthetic evidence sentence."
    assert set(item) == {"evidence_id", "source_id", "source_url", "source_type", "title", "author", "published_at",
                         "scraped_at", "last_scraped_at", "observed_at", "evidence_text", "segment_id",
                         "segment_kind", "segment_author", "segment_url", "segment_published_at",
                         "start_offset", "end_offset", "rule_id", "quality_tier"}
    assert item["author"] is None and item["published_at"] is None
    assert item["segment_author"] is None and item["segment_url"] is None  # segments_json has no such segment
    page = analysis.edge_sources(conn, edge, limit=1, offset=2)
    assert page["total"] == 3 and [i["observed_at"] for i in page["items"]] == [timestamp(T0 + timedelta(hours=2))]
    with pytest.raises(analysis.EdgeNotFound):
        analysis.edge_sources(conn, edge + 100)


def test_truncated_network_never_returns_a_node_without_an_edge_towards_the_root(conn):
    """Regression: with max_edges hit, same-level edges used to fill the budget first and left the depth-2 nodes of
    the most central entities (Iran, United States in the calibration graph) floating without any edge."""
    def fill(g):
        n = {name: g.node(name, mentions=m) for name, m in
             [("Root", 9), ("Alpha", 5), ("Bravo", 4), ("Charlie", 3), ("Delta", 2), ("Echo", 1)]}
        for a, b in [("Root", "Alpha"), ("Root", "Bravo"), ("Root", "Charlie"), ("Alpha", "Bravo"),
                     ("Alpha", "Charlie"), ("Bravo", "Charlie"), ("Alpha", "Delta"), ("Bravo", "Echo")]:
            g.edge(n[a], n[b])
        return n
    n = build(conn, fill)

    def check(result):
        distance = {node["id"]: node["distance"] for node in result["nodes"]}
        for node_id, d in distance.items():
            if d > 0:
                assert any({e["source"], e["target"]} == {node_id, other} for e in result["edges"]
                           for other, od in distance.items() if od == d - 1), node_id
        assert all(e["source"] in distance and e["target"] in distance for e in result["edges"])

    result = analysis.network(conn, "Root", max_edges=5)
    check(result)
    assert len(result["nodes"]) == 6 and len(result["edges"]) == 5
    assert result["meta"] == {"truncated": True, "reason": "max_edges", "max_nodes": 200, "max_edges": 5}
    # A budget smaller than the node count drops the lowest-ranked nodes rather than orphaning them.
    small = analysis.network(conn, "Root", max_edges=3)
    check(small)
    assert {node["name"] for node in small["nodes"]} == {"Root", "Alpha", "Bravo", "Charlie"}
    assert str(n["Delta"]) not in {node["id"] for node in small["nodes"]}


def test_edge_sources_attribute_comment_evidence_to_the_comment_not_the_thread(conn):
    """Synthetic thread: the submitter posted the story, a different user wrote the comment the edge comes from."""
    edge = build(conn, lambda g: g.edge(g.node("Ev C"), g.node("Ev D")))
    segments = [{"id": "story", "text": "Synthetic story title", "kind": "post", "author": "submitter",
                 "url": "https://forum.example/item?id=1", "published_at": "2026-01-01T00:00:00Z"},
                {"id": "c7", "text": "Synthetic evidence sentence.", "kind": "comment", "author": "commenter",
                 "parent_id": "story", "url": "https://forum.example/item?id=7", "published_at": "2026-01-02T08:00:00Z"}]
    with db.transaction(conn):
        conn.execute("UPDATE sources SET author = 'submitter', source_type = 'discussion', segments_json = ?",
                     (json.dumps(segments),))
        conn.execute("UPDATE edge_sources SET segment_id = 'c7' WHERE edge_id = ?", (edge,))
    item = analysis.edge_sources(conn, edge)["items"][0]
    assert item["author"] == "submitter"  # the thread
    assert (item["segment_kind"], item["segment_author"], item["segment_url"], item["segment_published_at"]) == (
        "comment", "commenter", "https://forum.example/item?id=7", "2026-01-02T08:00:00Z")


def test_queries_join_an_open_transaction_instead_of_ending_it(conn, sample):
    conn.execute("BEGIN")
    analysis.central(conn)
    analysis.network(conn, "Alice Example")
    assert conn.in_transaction
    conn.execute("ROLLBACK")
    assert analysis.graph_counts(conn) == {"nodes": 6, "edges": 6, "sources": 6, "evidence": 6}


def test_read_only_connection_sees_one_committed_snapshot(conn, sample, tmp_path):
    reader = db.connect(tmp_path / "graph.sqlite3", readonly=True)
    try:
        with db.transaction(conn):
            Graph(conn).node("Uncommitted While Reading")
            assert analysis.central(reader)["node_count"] == 6
        reader.execute("BEGIN")
        assert analysis.central(reader)["node_count"] == 7
        build(conn, lambda g: g.node("Committed During Snapshot"))
        assert analysis.central(reader)["node_count"] == 7
        reader.execute("ROLLBACK")
        assert analysis.central(reader)["node_count"] == 8
    finally:
        reader.close()
