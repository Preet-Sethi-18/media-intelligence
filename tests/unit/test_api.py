"""HTTP contract tests with FastAPI TestClient over a temporary SQLite file of small synthetic rows (not crawled data)."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from media_intelligence import analysis, api, db
from media_intelligence.config import Settings
from media_intelligence.models import node_key, normalize_name, timestamp, utcnow

T0 = datetime(2026, 1, 1, 12, tzinfo=UTC)
SINCE = datetime(2026, 2, 1, 12, tzinfo=UTC)


def add_node(conn, name, entity_type="PERSON", aliases=()):
    node_id = conn.execute(
        """INSERT INTO nodes (key, name, normalized_name, entity_type, first_seen, mention_count)
           VALUES (?, ?, ?, ?, ?, 1)""",
        (node_key(entity_type, name), name, normalize_name(name), entity_type, timestamp(T0))).lastrowid
    for alias in aliases:
        conn.execute("INSERT INTO node_aliases VALUES (?, ?, 'handle', 'test')", (normalize_name(alias), node_id))
    return node_id


def add_edge(conn, run, source, target, relation="met_with", directed=False, times=(T0,)):
    """Each observation time gets its own synthetic source URL."""
    if not directed:
        source, target = sorted((source, target))
    edge_id = conn.execute(
        """INSERT INTO edges (source_node_id, target_node_id, relation_type, directed, weight, first_seen, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (source, target, relation, int(directed), len(times), timestamp(min(times)), timestamp(max(times)))).lastrowid
    for index, observed in enumerate(times):
        url = f"https://example.com/{edge_id}/{index}"
        source_id = conn.execute(
            """INSERT INTO sources (source_url, requested_url, source_type, scraped_at, last_scraped_at, body,
               content_hash, crawl_run_id) VALUES (?, ?, 'news', ?, ?, 'Synthetic body.', ?, ?)""",
            (url, url, timestamp(observed), timestamp(observed), f"hash-{url}", run)).lastrowid
        conn.execute(
            """INSERT INTO edge_sources (edge_id, source_id, observed_at, evidence_text, segment_id, start_offset,
               end_offset, rule_id, quality_tier) VALUES (?, ?, ?, 'Synthetic.', 'body', 0, 10, 'test', 'cooccurrence')""",
            (edge_id, source_id, timestamp(observed)))
    return edge_id


@pytest.fixture
def graph(tmp_path):
    """Synthetic graph: Elon Musk (@elonmusk) - Space Agency -> Mars Topic, a new edge, and two 'Jordan' nodes."""
    path = tmp_path / "api.sqlite3"
    conn = db.connect(path)
    db.migrate(conn)
    with db.transaction(conn):
        run = conn.execute("INSERT INTO crawl_runs (started_at, config_hash, extractor_version) "
                           "VALUES (?, 'cfg', 'test')", (timestamp(T0),)).lastrowid
        ids = {"musk": add_node(conn, "Elon Musk", aliases=["@elonmusk", "Musk"]),
               "agency": add_node(conn, "Space Agency", "ORG"), "topic": add_node(conn, "Mars Topic", "TOPIC"),
               "jordan_person": add_node(conn, "Jordan"), "jordan_place": add_node(conn, "Jordan", "LOCATION")}
        ids["e1"] = add_edge(conn, run, ids["musk"], ids["agency"])
        ids["e2"] = add_edge(conn, run, ids["agency"], ids["topic"], "discussed_topic", True)
        ids["e3"] = add_edge(conn, run, ids["musk"], ids["jordan_place"], "criticized", True,
                             times=(SINCE + timedelta(hours=1),))
    conn.close()
    return path, ids


@pytest.fixture
def client(graph):
    return TestClient(api.create_app(Settings(database_path=graph[0])))


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"]["code"] == code and body["error"]["message"]
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]
    assert "Traceback" not in response.text
    return body["error"]


def test_network_depth_one_and_two_with_encoded_handle(client, graph):
    _, ids = graph
    one = client.get("/entity/%40elonmusk/network", params={"depth": 1})
    assert one.status_code == 200 and one.headers["X-Request-ID"]
    body = one.json()
    assert body["root_id"] == str(ids["musk"]) and body["depth"] == 1
    assert {n["id"] for n in body["nodes"]} == {str(ids[k]) for k in ("musk", "agency", "jordan_place")}
    assert body["meta"] == {"truncated": False, "reason": None, "max_nodes": 200, "max_edges": 500}
    two = client.get("/entity/Elon%20Musk/network").json()
    assert two["depth"] == 2 and {n["id"] for n in two["nodes"]} == {str(ids[k]) for k in (
        "musk", "agency", "jordan_place", "topic")}
    directed = next(e for e in two["edges"] if e["id"] == str(ids["e2"]))
    assert (directed["source"], directed["target"], directed["directed"], directed["weight"]) == (
        str(ids["agency"]), str(ids["topic"]), True, 1)
    assert directed["evidence_url"] == f"/edges/{ids['e2']}/sources"


def test_network_truncation_flag(client):
    body = client.get("/entity/musk/network", params={"max_nodes": 2, "max_edges": 1}).json()
    assert len(body["nodes"]) == 2 and len(body["edges"]) == 1
    assert body["meta"]["truncated"] is True and body["meta"]["reason"] == "max_nodes"


def test_ambiguous_name_returns_409_with_candidates_and_entity_id_selects(client, graph):
    _, ids = graph
    error = assert_error(client.get("/entity/jordan/network"), 409, "ambiguous_entity")
    assert error["candidates"] == [
        {"id": str(ids["jordan_person"]), "name": "Jordan", "type": "PERSON", "mention_count": 1},
        {"id": str(ids["jordan_place"]), "name": "Jordan", "type": "LOCATION", "mention_count": 1}]
    chosen = client.get("/entity/Jordan/network", params={"entity_id": ids["jordan_place"], "depth": 1})
    assert chosen.status_code == 200 and chosen.json()["root_id"] == str(ids["jordan_place"])
    assert_error(client.get("/entity/Jordan/network", params={"entity_id": ids["musk"]}), 404, "entity_not_found")


def test_unknown_entity_and_edge_return_404(client):
    assert_error(client.get("/entity/Nobody%20Here/network"), 404, "entity_not_found")
    assert_error(client.get("/edges/9999/sources"), 404, "edge_not_found")
    assert_error(client.get("/no/such/route"), 404, "not_found")


@pytest.mark.parametrize("url", [
    "/entity/Elon%20Musk/network?depth=3", "/entity/Elon%20Musk/network?depth=0",
    "/entity/Elon%20Musk/network?max_nodes=201", "/entity/Elon%20Musk/network?max_edges=501",
    "/entity/%20%20/network", "/entity/" + "x" * 201 + "/network", "/entities/central?limit=101",
    "/entities/central?limit=0", "/entities/central?offset=-1", "/edges/abc/sources", "/edges/1/sources?limit=500",
    "/connections/new", "/connections/new?since=yesterday", "/connections/new?since=2026-10-07T12:00:00",
    "/connections/new?since=2026-02-01", "/connections/new?since=2026-02-01T12:00:00Z&offset=-1",
])
def test_invalid_parameters_return_422_envelope_with_details(client, url):
    error = assert_error(client.get(url), 422, "validation_error")
    assert error["details"] and all({"loc", "msg", "type"} <= set(item) for item in error["details"])


def test_naive_and_future_since_are_rejected(client):
    naive = assert_error(client.get("/connections/new", params={"since": "2026-02-01T12:00:00"}),
                         422, "validation_error")
    assert naive["details"][0]["type"] == "timezone_required"
    future = timestamp(utcnow() + timedelta(days=1))
    error = assert_error(client.get("/connections/new", params={"since": future}), 422, "validation_error")
    assert error["details"][0]["type"] == "datetime_future"


def test_new_connections_envelope_and_offset_equivalence(client, graph):
    _, ids = graph
    response = client.get("/connections/new", params={"since": "2026-02-01T12:00:00Z"})
    assert response.status_code == 200
    body = response.json()
    assert body["since"] == timestamp(SINCE) and body["as_of"] > body["since"]
    assert body["thresholds"] == {"min_delta": 3, "min_ratio": 0.5}
    assert (body["total"], body["limit"], body["offset"]) == (1, 20, 0)
    item = body["items"][0]
    assert (item["edge_id"], item["reason"], item["weight_before"], item["weight_now"], item["weight_delta"]) == (
        str(ids["e3"]), "new", 0, 1, 1)
    assert item["relative_growth"] is None and item["source"]["name"] == "Elon Musk"
    shifted = client.get("/connections/new?since=2026-02-01T17:30:00%2B05:30").json()
    assert shifted["since"] == body["since"] and shifted["items"] == body["items"]
    # A hand-typed, unencoded "+" arrives as a space; it is read back as the offset it was.
    for raw in ("2026-02-01T17:30:00+05:30", "2026-02-01T12:00:00+00:00", "2026-02-01T17:30:00+0530"):
        unencoded = client.get(f"/connections/new?since={raw}")
        assert unencoded.status_code == 200 and unencoded.json()["since"] == body["since"], raw
    later = client.get("/connections/new", params={"since": "2026-02-01T13:00:00.000001Z"}).json()
    assert later["items"] == [] and later["total"] == 0


def test_thresholds_come_from_settings(graph):
    settings = Settings(database_path=graph[0], growth_min_delta=5, growth_min_ratio=0.25)
    body = TestClient(api.create_app(settings)).get("/connections/new", params={"since": "2026-01-01T00:00:00Z"})
    assert body.json()["thresholds"] == {"min_delta": 5, "min_ratio": 0.25}


def test_central_and_evidence_routes(client, graph):
    _, ids = graph
    central = client.get("/entities/central", params={"limit": 2}).json()
    assert (central["metric"], central["node_count"], central["total"], central["limit"]) == (
        "normalized_degree", 5, 5, 2)
    assert [(i["rank"], i["node_id"], i["score"]) for i in central["items"]] == [
        (1, str(ids["musk"]), 0.5), (2, str(ids["agency"]), 0.5)]
    evidence = client.get(f"/edges/{ids['e1']}/sources").json()
    assert evidence["total"] == 1 and evidence["edge"]["id"] == str(ids["e1"])
    assert evidence["items"][0]["evidence_text"] == "Synthetic." and "body" not in evidence["items"][0]


def test_health_reports_version_schema_and_counts(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "version": api.__version__, "schema_version": db.SCHEMA_VERSION,
                    "counts": {"nodes": 5, "edges": 3, "sources": 3, "evidence": 3}}


def test_missing_or_unmigrated_database_returns_503_without_creating_files(tmp_path):
    missing = tmp_path / "missing.sqlite3"
    client = TestClient(api.create_app(Settings(database_path=missing)))
    for url in ("/health", "/entities/central", "/entity/x/network", "/connections/new?since=2026-01-01T00:00:00Z"):
        error = assert_error(client.get(url), 503, "database_unavailable")
        assert str(tmp_path) not in error["message"]
    assert not missing.exists()
    empty = tmp_path / "empty.sqlite3"
    db.connect(empty).close()
    assert_error(TestClient(api.create_app(Settings(database_path=empty))).get("/health"), 503,
                 "database_unavailable")
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"not a database" * 100)
    assert_error(TestClient(api.create_app(Settings(database_path=garbage))).get("/health"), 503,
                 "database_unavailable")


def test_unreadable_database_file_or_folder_returns_503(graph, tmp_path):
    path = graph[0]
    client = TestClient(api.create_app(Settings(database_path=path)))
    for target in (path, path.parent):
        mode = target.stat().st_mode
        target.chmod(0)
        try:
            error = assert_error(client.get("/health"), 503, "database_unavailable")
        finally:
            target.chmod(mode)
        assert str(path) not in error["message"]
    assert client.get("/health").status_code == 200


def test_unexpected_error_returns_500_envelope_without_traceback(client, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(analysis, "central", broken)
    safe = TestClient(client.app, raise_server_exceptions=False)
    response = safe.get("/entities/central")
    error = assert_error(response, 500, "internal_error")
    assert "secret" not in response.text and "details" not in error


def test_request_id_is_echoed_when_safe_and_replaced_otherwise(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    replaced = client.get("/entity/Nobody/network", headers={"X-Request-ID": "bad id!"})
    assert replaced.headers["X-Request-ID"] != "bad id!" and len(replaced.headers["X-Request-ID"]) == 32


def test_api_has_no_write_routes(client):
    methods = {method for route in client.app.routes for method in getattr(route, "methods", ())}
    assert methods <= {"GET", "HEAD"}
    assert_error(client.post("/entities/central"), 405, "method_not_allowed")
