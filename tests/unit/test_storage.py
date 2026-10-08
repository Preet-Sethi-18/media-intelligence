"""Write-path tests on an in-memory database; every item, mention and relation here is labelled synthetic data."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from media_intelligence import db
from media_intelligence.models import ContentItem, EntityMention, Extraction, Relation, Segment, node_key, timestamp
from media_intelligence.storage import (
    StoreResult,
    finish_run,
    graph_counts,
    source_version_id,
    start_run,
    store_item,
    verify_integrity,
)

T0 = datetime(2026, 10, 7, 12, tzinfo=UTC)
T1 = T0 + timedelta(hours=6)
T2 = T0 + timedelta(days=1)
URL = "https://news.example.com/synthetic-story"
URL2 = "https://other.example.org/synthetic-report"

# Synthetic article text; it is not crawled content.
TEXT = "Elon Musk met Narendra Modi in Delhi. Musk criticized the UN. @elonmusk later posted about Delhi."
MUSK, MODI = node_key("PERSON", "Elon Musk"), node_key("PERSON", "Narendra Modi")
DELHI, UN = node_key("LOCATION", "Delhi"), node_key("ORG", "United Nations")
TABLES = ("crawl_runs", "sources", "nodes", "edges", "edge_sources", "node_mentions", "node_aliases")


@pytest.fixture
def conn():
    connection = db.connect(":memory:")
    db.migrate(connection)
    yield connection
    connection.close()


@pytest.fixture
def run_id(conn):
    return start_run(conn, config_hash="cfg", extractor_version="test", started_at=T0)


def item(text=TEXT, url=URL, **fields):
    return ContentItem(source_url=url, source_type="news", scraped_at=T0, body=text,
                       segments=[Segment(id="body", text=text)], **fields)


def mention(surface, canonical, entity_type="PERSON", resolution="exact", text=TEXT, after=0):
    start = text.index(surface, after)
    return EntityMention(name=surface, entity_type=entity_type, canonical_name=canonical,
                         key=node_key(entity_type, canonical), segment_id="body", start=start,
                         end=start + len(surface), resolution=resolution)


def relation(source, target, sentence, relation_type="criticized", directed=True, tier="semantic_rule", text=TEXT):
    start = text.index(sentence)
    return Relation(source_key=source, target_key=target, relation_type=relation_type, directed=directed,
                    evidence_text=sentence, segment_id="body", start=start, end=start + len(sentence),
                    rule_id=f"{relation_type}_rule", quality_tier=tier)


def mentions(text=TEXT):
    return [mention("Elon Musk", "Elon Musk", text=text),
            mention("Narendra Modi", "Narendra Modi", text=text),
            mention("Delhi", "Delhi", "LOCATION", text=text),
            mention("Musk", "Elon Musk", resolution="surname", text=text, after=text.index(".")),
            mention("UN", "United Nations", "ORG", resolution="abbreviation", text=text),
            mention("@elonmusk", "Elon Musk", resolution="handle", text=text),
            mention("Delhi", "Delhi", "LOCATION", text=text, after=text.index("@"))]


def extraction(text=TEXT):
    return Extraction(mentions=mentions(text), relations=[
        relation(MUSK, MODI, "Elon Musk met Narendra Modi in Delhi.", "met_with", directed=False, text=text),
        relation(MUSK, UN, "Musk criticized the UN.", text=text),
        relation(MODI, DELHI, "Elon Musk met Narendra Modi in Delhi.", "mentioned_with", False, "cooccurrence",
                 text=text)])


def rows(conn, sql, *params):
    return [tuple(row) for row in conn.execute(sql, params).fetchall()]


def snapshot(conn, ignore=()):
    return {table: [{k: v for k, v in dict(row).items() if k not in ignore}
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")] for table in TABLES}


def edge(conn, source_key, target_key, relation_type):
    return conn.execute("""SELECT e.* FROM edges e JOIN nodes a ON a.id = e.source_node_id
                           JOIN nodes b ON b.id = e.target_node_id
                           WHERE a.key = ? AND b.key = ? AND e.relation_type = ?""",
                        (source_key, target_key, relation_type)).fetchone()


def test_new_item_stores_source_nodes_mentions_aliases_edges_and_evidence(conn, run_id):
    content = item(title="Synthetic talks", published_at=T0 - timedelta(days=1), metadata={"votes": 3})
    result = store_item(conn, content, extraction(), run_id=run_id, observed_at=T1)
    assert result == StoreResult(source_id=1, new_version=True, nodes_created=4, edges_created=3,
                                 evidence_added=3, mentions_added=4)

    source = conn.execute("SELECT * FROM sources").fetchone()
    assert source["source_url"] == source["requested_url"] == URL
    assert source["scraped_at"] == source["last_scraped_at"] == timestamp(T1)
    assert source["published_at"] == timestamp(T0 - timedelta(days=1))
    assert source["content_hash"] == content.content_hash and source["crawl_run_id"] == run_id
    assert json.loads(source["segments_json"])[0]["text"] == TEXT
    assert json.loads(source["metadata_json"]) == {"votes": 3}

    assert rows(conn, "SELECT key, name, normalized_name, entity_type, first_seen, mention_count FROM nodes") == [
        (MUSK, "Elon Musk", "elon musk", "PERSON", timestamp(T1), 1),
        (MODI, "Narendra Modi", "narendra modi", "PERSON", timestamp(T1), 1),
        (DELHI, "Delhi", "delhi", "LOCATION", timestamp(T1), 1),
        (UN, "United Nations", "united nations", "ORG", timestamp(T1), 1)]
    assert rows(conn, "SELECT node_id, surface_forms_json, occurrence_count, first_observed_at FROM node_mentions") == [
        (1, '["@elonmusk", "Elon Musk", "Musk"]', 3, timestamp(T1)), (2, '["Narendra Modi"]', 1, timestamp(T1)),
        (3, '["Delhi"]', 2, timestamp(T1)), (4, '["UN"]', 1, timestamp(T1))]
    assert rows(conn, "SELECT alias_key, node_id, alias_kind, origin FROM node_aliases ORDER BY alias_key") == [
        ("@elonmusk", 1, "handle", "observed"), ("delhi", 3, "surface", "observed"),
        ("elon musk", 1, "surface", "observed"), ("musk", 1, "surname", "observed"),
        ("narendra modi", 2, "surface", "observed"), ("un", 4, "abbreviation", "observed")]

    criticized = edge(conn, MUSK, UN, "criticized")
    assert (criticized["directed"], criticized["weight"]) == (1, 1)
    assert criticized["first_seen"] == criticized["last_seen"] == timestamp(T1)
    evidence = conn.execute("SELECT * FROM edge_sources WHERE edge_id = ?", (criticized["id"],)).fetchone()
    assert TEXT[evidence["start_offset"]:evidence["end_offset"]] == evidence["evidence_text"] == \
        "Musk criticized the UN."
    assert (evidence["source_id"], evidence["observed_at"], evidence["segment_id"], evidence["rule_id"],
            evidence["quality_tier"]) == (1, timestamp(T1), "body", "criticized_rule", "semantic_rule")
    assert not conn.in_transaction
    assert verify_integrity(conn) == []


def test_identical_rerun_changes_no_counts_and_only_refreshes_recency(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    rerun = start_run(conn, config_hash="cfg", extractor_version="test", started_at=T2)
    assert source_version_id(conn, item()) == 1
    recency = ("last_scraped_at", "last_seen")
    before = snapshot(conn, ignore=recency)

    result = store_item(conn, item(), extraction(), run_id=rerun, observed_at=T2)
    assert result == StoreResult(source_id=1, new_version=False)
    assert snapshot(conn, ignore=recency) == before
    after = snapshot(conn)
    assert rows(conn, "SELECT scraped_at, last_scraped_at FROM sources") == [(timestamp(T0), timestamp(T2))]
    assert {r["last_seen"] for r in after["edges"]} == {timestamp(T2)}
    assert {r["first_seen"] for r in after["edges"]} == {timestamp(T0)}

    # An older reobservation never moves recency backwards, and a duplicate ignores whatever extraction is passed.
    refreshed = snapshot(conn)
    assert store_item(conn, item(), Extraction(), run_id=rerun, observed_at=T1).new_version is False
    assert snapshot(conn) == refreshed
    assert verify_integrity(conn) == []


def test_new_version_of_same_url_adds_evidence_without_weight(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    edited = TEXT + " Officials later confirmed the meeting."
    result = store_item(conn, item(edited), extraction(edited), run_id=run_id, observed_at=T2)
    assert (result.new_version, result.nodes_created, result.edges_created) == (True, 0, 0)
    assert (result.evidence_added, result.mentions_added) == (3, 4)

    criticized = edge(conn, MUSK, UN, "criticized")
    assert criticized["weight"] == 1
    assert (criticized["first_seen"], criticized["last_seen"]) == (timestamp(T0), timestamp(T2))
    assert rows(conn, "SELECT source_id, observed_at FROM edge_sources WHERE edge_id = ? ORDER BY id",
                criticized["id"]) == [(1, timestamp(T0)), (2, timestamp(T2))]
    assert rows(conn, "SELECT mention_count, first_seen FROM nodes WHERE key = ?", MUSK) == [(1, timestamp(T0))]
    assert rows(conn, "SELECT count(*), count(DISTINCT source_url) FROM sources") == [(2, 1)]
    assert verify_integrity(conn) == []


def test_second_url_increases_weight_and_document_frequency(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    other = "Musk criticized the UN again."
    extra = Extraction(mentions=[mention("Musk", "Elon Musk", resolution="surname", text=other),
                                 mention("UN", "United Nations", "ORG", resolution="abbreviation", text=other)],
                       relations=[relation(MUSK, UN, other, text=other)])
    result = store_item(conn, item(other, url=URL2), extra, run_id=run_id, observed_at=T1)
    assert (result.nodes_created, result.edges_created, result.evidence_added) == (0, 0, 1)

    criticized = edge(conn, MUSK, UN, "criticized")
    assert criticized["weight"] == 2
    assert (criticized["first_seen"], criticized["last_seen"]) == (timestamp(T0), timestamp(T1))
    assert rows(conn, "SELECT key, mention_count FROM nodes ORDER BY id") == [(MUSK, 2), (MODI, 1), (DELHI, 1), (UN, 2)]
    assert edge(conn, MUSK, MODI, "met_with")["weight"] == 1
    assert verify_integrity(conn) == []


def test_symmetric_relations_use_ascending_node_ids_and_directed_keep_direction(conn, run_id):
    reversed_pair = Extraction(mentions=mentions(), relations=[
        relation(DELHI, MODI, "Elon Musk met Narendra Modi in Delhi.", "mentioned_with", False, "cooccurrence"),
        relation(MODI, MUSK, "Elon Musk met Narendra Modi in Delhi.", "met_with", directed=False),
        relation(MUSK, MODI, "Musk criticized the UN.", "met_with", directed=False),
        relation(UN, MUSK, "Musk criticized the UN."),
        relation(MUSK, UN, "Musk criticized the UN.")])
    result = store_item(conn, item(), reversed_pair, run_id=run_id, observed_at=T0)
    assert (result.edges_created, result.evidence_added) == (4, 4)
    assert rows(conn, """SELECT a.key, b.key, e.relation_type, e.directed, e.weight FROM edges e
                         JOIN nodes a ON a.id = e.source_node_id JOIN nodes b ON b.id = e.target_node_id
                         ORDER BY e.id""") == [
        (MODI, DELHI, "mentioned_with", 0, 1), (MUSK, MODI, "met_with", 0, 1),
        (UN, MUSK, "criticized", 1, 1), (MUSK, UN, "criticized", 1, 1)]
    # Both met_with orderings collapse to one edge; the first relation's excerpt is the representative evidence.
    met = edge(conn, MUSK, MODI, "met_with")
    assert rows(conn, "SELECT evidence_text FROM edge_sources WHERE edge_id = ?", met["id"]) == [
        ("Elon Musk met Narendra Modi in Delhi.",)]
    assert verify_integrity(conn) == []


def test_self_edges_missing_endpoints_and_invalid_evidence_are_rejected(conn, run_id):
    sentence = "Musk criticized the UN."
    bad_offsets = relation(MUSK, UN, sentence).model_copy(update={"start": 1})
    unknown_segment = relation(MUSK, UN, sentence).model_copy(update={"segment_id": "comment-9"})
    candidate = Extraction(mentions=mentions(), relations=[
        relation(MUSK, MUSK, "Elon Musk met Narendra Modi in Delhi.", "met_with", directed=False),
        relation(MUSK, node_key("PERSON", "Elon"), sentence),
        relation(node_key("ORG", "Musk"), UN, sentence),
        bad_offsets, unknown_segment])
    result = store_item(conn, item(), candidate, run_id=run_id, observed_at=T0)
    assert (result.self_edges_rejected, result.relations_skipped, result.invalid_evidence) == (1, 2, 2)
    assert (result.edges_created, result.evidence_added, result.nodes_created) == (0, 0, 4)
    assert rows(conn, "SELECT count(*) FROM edges") == [(0,)]
    assert verify_integrity(conn) == []


def test_mention_with_inconsistent_key_is_refused_before_writing(conn, run_id):
    wrong = mention("Musk", "Elon Musk").model_copy(update={"key": "PERSON:musk"})
    before = snapshot(conn)
    with pytest.raises(ValueError, match="does not match"):
        store_item(conn, item(), Extraction(mentions=[wrong]), run_id=run_id, observed_at=T0)
    assert snapshot(conn) == before


def test_failure_mid_transaction_rolls_back_everything(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    before = snapshot(conn)
    other = "Narendra Modi criticized the UN."
    second = Extraction(mentions=[mention("Narendra Modi", "Narendra Modi", text=other),
                                  mention("UN", "United Nations", "ORG", "abbreviation", text=other)],
                        relations=[relation(MODI, UN, other, text=other)])
    # Fails after the source, node mentions, aliases and edge rows were written inside the item transaction.
    conn.execute("""CREATE TEMP TRIGGER injected_failure BEFORE INSERT ON edge_sources
                    BEGIN SELECT RAISE(ABORT, 'injected failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        store_item(conn, item(other, url=URL2), second, run_id=run_id, observed_at=T1)
    assert not conn.in_transaction
    assert snapshot(conn) == before

    conn.execute("DROP TRIGGER injected_failure")
    assert store_item(conn, item(other, url=URL2), second, run_id=run_id, observed_at=T1).evidence_added == 1
    assert edge(conn, MODI, UN, "criticized")["weight"] == 1
    assert verify_integrity(conn) == []


def test_python_error_mid_transaction_rolls_back_everything(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    before = snapshot(conn)
    other = "Elon Musk met Narendra Modi in Paris."
    conflicting = Extraction(
        mentions=[mention("Elon Musk", "Elon Musk", text=other), mention("Narendra Modi", "Narendra Modi", text=other),
                  mention("Paris", "Paris", "LOCATION", text=other)],
        relations=[relation(MODI, node_key("LOCATION", "Paris"), other, "mentioned_with", False, "cooccurrence",
                            text=other),
                   relation(MUSK, MODI, other, "met_with", directed=True, text=other)])
    with pytest.raises(ValueError, match="directed"):
        store_item(conn, item(other, url=URL2), conflicting, run_id=run_id, observed_at=T1)
    assert not conn.in_transaction
    assert snapshot(conn) == before


def test_verify_integrity_reports_corruption(conn, run_id):
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    assert verify_integrity(conn) == []
    conn.execute("UPDATE edges SET weight = 5 WHERE relation_type = 'criticized'")
    conn.execute("UPDATE nodes SET mention_count = 3 WHERE key = ?", (UN,))
    conn.execute("""INSERT INTO edges (source_node_id, target_node_id, relation_type, directed, weight, first_seen,
                    last_seen) VALUES (3, 4, 'located_in', 1, 1, ?, ?)""", (timestamp(T0), timestamp(T0)))
    conn.execute("UPDATE sources SET last_scraped_at = ?", (timestamp(T2),))
    problems = verify_integrity(conn)
    # The evidence-less edge also has a weight with no supporting URL behind it.
    assert "edge_weight_mismatches: 2" in problems
    assert "node_mention_count_mismatches: 1" in problems
    assert "edges_without_evidence: 1" in problems
    assert "edge_last_seen_stale: 3" in problems

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("UPDATE node_aliases SET node_id = 99 WHERE alias_key = 'un'")
    assert "foreign_key_violations: 1" in verify_integrity(conn)


def test_graph_counts_for_run_reports(conn, run_id):
    assert graph_counts(conn)["nodes"] == 0
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0)
    edited = TEXT + " Synthetic edit."
    store_item(conn, item(edited), extraction(edited), run_id=run_id, observed_at=T1)
    assert graph_counts(conn) == {
        "sources": 2, "source_urls": 1, "nodes": 4, "nodes_by_type": {"LOCATION": 1, "ORG": 1, "PERSON": 2},
        "aliases": 6, "edges": 3, "edges_by_relation_type": {"criticized": 1, "mentioned_with": 1, "met_with": 1},
        "edges_by_quality_tier": {"cooccurrence": 1, "semantic_rule": 2}, "evidence": 6,
        "evidence_by_quality_tier": {"cooccurrence": 2, "semantic_rule": 4},
        "evidence_by_source_type_and_tier": {"news": {"cooccurrence": 2, "semantic_rule": 4}},
        "semantic_evidence_by_source_type_and_relation": {"news": {"criticized": 2, "met_with": 2}}}


def test_run_lifecycle(conn, run_id):
    assert rows(conn, "SELECT status, finished_at, started_at FROM crawl_runs") == [("running", None, timestamp(T0))]
    finish_run(conn, run_id, status="partial", summary={"failed": {"x": 2}, "ok": 5}, finished_at=T1)
    run = conn.execute("SELECT * FROM crawl_runs WHERE id = ?", (run_id,)).fetchone()
    assert (run["status"], run["finished_at"]) == ("partial", timestamp(T1))
    assert json.loads(run["summary_json"]) == {"failed": {"x": 2}, "ok": 5}
    with pytest.raises(ValueError, match="status"):
        finish_run(conn, run_id, status="running", summary={}, finished_at=T1)
    with pytest.raises(ValueError, match="Unknown"):
        finish_run(conn, 999, status="complete", summary={}, finished_at=T1)
    with pytest.raises(ValueError, match="timezone"):
        start_run(conn, config_hash="cfg", extractor_version="test", started_at=datetime(2026, 10, 7))  # noqa: DTZ001


def test_reviewed_aliases_are_stored_for_touched_nodes_with_their_reason(conn, run_id):
    reviewed = {MUSK: [("@elonmusk", "handle"), ("elonmusk", "handle"), ("musk", "alias")],
                UN: [("un", "abbreviation"), ("u.n.", "abbreviation")],
                node_key("PERSON", "Synthetic Absent Person"): [("absent", "alias")]}
    store_item(conn, item(), extraction(), run_id=run_id, observed_at=T0, reviewed_aliases=reviewed)
    aliases = {(a, k, o) for a, k, o in rows(conn, """SELECT alias_key, alias_kind, origin FROM node_aliases
                                                    JOIN nodes ON nodes.id = node_id""")}
    assert {("elonmusk", "handle", "reviewed"), ("u.n.", "abbreviation", "reviewed"),
            ("musk", "alias", "reviewed")} <= aliases
    assert ("@elonmusk", "handle", "reviewed") in aliases  # a reviewed row wins over the same observed form
    assert not any(alias == "absent" for alias, _, _ in aliases)  # nodes not in this item get nothing
    assert verify_integrity(conn) == []


def test_observed_reviewed_alias_resolution_keeps_its_own_kind(conn, run_id):
    text = "Putin met the press."
    alias = EntityMention(name="Putin", entity_type="PERSON", canonical_name="Vladimir Putin",
                          key=node_key("PERSON", "Vladimir Putin"), segment_id="body", start=0, end=5,
                          resolution="alias")
    store_item(conn, item(text), Extraction(mentions=[alias]), run_id=run_id, observed_at=T0)
    assert rows(conn, "SELECT alias_key, alias_kind, origin FROM node_aliases") == [("putin", "alias", "observed")]


def test_default_observation_time_is_the_store_clock_not_the_fetch_time(conn, run_id, monkeypatch):
    from media_intelligence import storage

    monkeypatch.setattr(storage, "utcnow", lambda: T1)  # synthetic clock: stored six hours after the fetch (T0)
    store_item(conn, item(), extraction(), run_id=run_id)
    assert rows(conn, "SELECT scraped_at, last_scraped_at FROM sources") == [(timestamp(T0), timestamp(T0))]
    assert {r[0] for r in conn.execute("SELECT observed_at FROM edge_sources")} == {timestamp(T1)}
    assert {r[0] for r in conn.execute("SELECT first_observed_at FROM node_mentions")} == {timestamp(T1)}
    assert {r[0] for r in conn.execute("SELECT first_seen FROM edges")} == {timestamp(T1)}
    # A clock behind the fetch (skew) never stamps an observation before the scrape.
    monkeypatch.setattr(storage, "utcnow", lambda: T0 - timedelta(hours=1))
    store_item(conn, item(url=URL2), extraction(), run_id=run_id)
    assert rows(conn, "SELECT min(observed_at) FROM edge_sources es JOIN sources s ON s.id = es.source_id "
                      "WHERE s.source_url = ?", URL2) == [(timestamp(T0),)]
    assert verify_integrity(conn) == []
