"""SQLite schema, migration, and transaction tests on temporary files, using small labelled synthetic rows."""

import sqlite3
from datetime import UTC, datetime

import pytest

from media_intelligence import db
from media_intelligence.models import node_key, normalize_name, timestamp

T0 = timestamp(datetime(2026, 10, 7, 12, tzinfo=UTC))
T1 = timestamp(datetime(2026, 10, 8, 12, tzinfo=UTC))

COLUMNS = {
    "nodes": ["id", "key", "name", "normalized_name", "entity_type", "first_seen", "mention_count"],
    "edges": ["id", "source_node_id", "target_node_id", "relation_type", "directed", "weight",
              "first_seen", "last_seen"],
    "sources": ["id", "source_url", "requested_url", "source_type", "scraped_at", "last_scraped_at", "title",
                "body", "author", "published_at", "content_hash", "segments_json", "metadata_json", "crawl_run_id"],
    "edge_sources": ["id", "edge_id", "source_id", "observed_at", "evidence_text", "segment_id", "start_offset",
                     "end_offset", "rule_id", "quality_tier"],
    "node_mentions": ["node_id", "source_id", "first_observed_at", "surface_forms_json", "occurrence_count"],
    "node_aliases": ["alias_key", "node_id", "alias_kind", "origin"],
    "crawl_runs": ["id", "started_at", "finished_at", "status", "config_hash", "extractor_version", "summary_json"],
}


# Synthetic rows below are hand-made test data, not crawled content.
def add_run(conn):
    return conn.execute("INSERT INTO crawl_runs (started_at, config_hash, extractor_version) VALUES (?, 'cfg', 'test')",
                        (T0,)).lastrowid


def add_source(conn, run_id, url="https://example.com/a", content_hash="hash-a"):
    return conn.execute(
        """INSERT INTO sources (source_url, requested_url, source_type, scraped_at, last_scraped_at, body,
           content_hash, crawl_run_id) VALUES (?, ?, 'news', ?, ?, 'Synthetic body text.', ?, ?)""",
        (url, url, T0, T0, content_hash, run_id)).lastrowid


def add_node(conn, name, entity_type="PERSON", mention_count=1):
    return conn.execute(
        """INSERT INTO nodes (key, name, normalized_name, entity_type, first_seen, mention_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (node_key(entity_type, name), name, normalize_name(name), entity_type, T0, mention_count)).lastrowid


def add_edge(conn, source, target, relation="MET_WITH", directed=1, weight=1):
    return conn.execute(
        """INSERT INTO edges (source_node_id, target_node_id, relation_type, directed, weight, first_seen, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?)""", (source, target, relation, directed, weight, T0, T0)).lastrowid


def add_evidence(conn, edge_id, source_id, start=0, end=9, tier="semantic_rule"):
    return conn.execute(
        """INSERT INTO edge_sources (edge_id, source_id, observed_at, evidence_text, segment_id, start_offset,
           end_offset, rule_id, quality_tier) VALUES (?, ?, ?, 'Synthetic', 'body', ?, ?, 'test_rule', ?)""",
        (edge_id, source_id, T0, start, end, tier)).lastrowid


def add_mention(conn, node_id, source_id, count=1):
    conn.execute("""INSERT INTO node_mentions (node_id, source_id, first_observed_at, surface_forms_json,
                    occurrence_count) VALUES (?, ?, ?, '["Synthetic"]', ?)""", (node_id, source_id, T0, count))


def add_alias(conn, alias, node_id):
    conn.execute("INSERT INTO node_aliases (alias_key, node_id, alias_kind, origin) VALUES (?, ?, 'abbreviation', "
                 "'test')", (normalize_name(alias), node_id))


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "graph.sqlite3")
    db.migrate(connection)
    yield connection
    connection.close()


@pytest.fixture
def graph(conn):
    """A synthetic one-edge graph that satisfies every invariant."""
    with db.transaction(conn):
        run = add_run(conn)
        source = add_source(conn, run)
        alice, acme = add_node(conn, "Alice Example"), add_node(conn, "Acme Corp", "ORG")
        edge = add_edge(conn, alice, acme, "WORKS_FOR")
        add_evidence(conn, edge, source)
        add_mention(conn, alice, source)
        add_mention(conn, acme, source)
    return conn, {"run": run, "source": source, "alice": alice, "acme": acme, "edge": edge}


def test_migrates_empty_database_and_reopens_read_only(tmp_path):
    path = tmp_path / "nested" / "graph.sqlite3"
    conn = db.connect(path)
    with pytest.raises(db.SchemaError):
        db.check_schema(conn)
    db.migrate(conn)
    db.check_schema(conn)
    assert db.user_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    for table, columns in COLUMNS.items():
        assert [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")] == columns
    with db.transaction(conn):
        add_node(conn, "Alice Example")
    conn.close()

    reader = db.connect(path, readonly=True)
    db.check_schema(reader)
    assert reader.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert reader.execute("SELECT key FROM nodes").fetchone()["key"] == "PERSON:alice example"
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        add_node(reader, "Bob Example")
    reader.close()


def test_expected_indexes_exist(conn):
    indexes = {row["name"]: row["tbl_name"] for row in conn.execute("SELECT name, tbl_name FROM sqlite_master "
                                                                    "WHERE type = 'index'")}
    assert {"edges_target_node", "edges_relation_type", "nodes_normalized_name", "sources_content_hash",
            "edge_sources_edge_observed", "edge_sources_source", "node_mentions_source"} <= set(indexes)
    plan = " ".join(row[3] for row in conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM edge_sources WHERE edge_id = 1 ORDER BY observed_at"))
    assert "edge_sources_edge_observed" in plan


def test_remigration_is_idempotent(graph):
    conn, _ = graph
    db.migrate(conn)
    db.migrate(conn)
    assert db.user_version(conn) == db.SCHEMA_VERSION
    assert conn.execute("SELECT count(*) FROM nodes").fetchone()[0] == 2


def test_refuses_newer_schema(conn):
    conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    with pytest.raises(db.SchemaError, match="newer"):
        db.migrate(conn)
    with pytest.raises(db.SchemaError):
        db.check_schema(conn)
    assert db.user_version(conn) == db.SCHEMA_VERSION + 1
    assert not conn.in_transaction


def test_failed_migration_rolls_back_whole_step(conn, monkeypatch):
    # Synthetic second migration: a trigger body and a string literal both contain semicolons.
    good = """CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT NOT NULL DEFAULT 'a;b');
        CREATE TRIGGER notes_guard BEFORE DELETE ON notes BEGIN SELECT RAISE(ABORT, 'keep; notes'); END;"""
    monkeypatch.setattr(db, "MIGRATIONS", (*db.MIGRATIONS, good + "\nINSERT INTO missing_table VALUES (1);"))
    monkeypatch.setattr(db, "SCHEMA_VERSION", len(db.MIGRATIONS))
    with pytest.raises(sqlite3.OperationalError, match="missing_table"):
        db.migrate(conn)
    assert db.user_version(conn) == db.SCHEMA_VERSION - 1
    assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name LIKE 'notes%'").fetchone()[0] == 0

    monkeypatch.setattr(db, "MIGRATIONS", (*db.MIGRATIONS[:-1], good))
    db.migrate(conn)
    db.check_schema(conn)
    conn.execute("INSERT INTO notes DEFAULT VALUES")
    assert conn.execute("SELECT text FROM notes").fetchone()[0] == "a;b"
    with pytest.raises(sqlite3.IntegrityError, match="keep; notes"):
        conn.execute("DELETE FROM notes")


def test_read_only_missing_or_invalid_file(tmp_path):
    missing = tmp_path / "absent" / "graph.sqlite3"
    with pytest.raises(db.SchemaError, match="not found"):
        db.connect(missing, readonly=True)
    assert not missing.parent.exists()

    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"synthetic non-database bytes" * 10)
    reader = db.connect(garbage, readonly=True)
    with pytest.raises(db.SchemaError, match="unreadable"):
        db.check_schema(reader)
    reader.close()
    with pytest.raises(db.SchemaError):
        db.connect(garbage)


def test_read_only_unreadable_file_is_a_schema_error(tmp_path):
    path = tmp_path / "graph.sqlite3"
    db.connect(path).close()
    path.chmod(0)
    try:
        with pytest.raises(db.SchemaError, match="Cannot open"):
            db.connect(path, readonly=True)
    finally:
        path.chmod(0o644)


def test_foreign_keys_are_enforced(graph):
    conn, ids = graph
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_edge(conn, ids["alice"], 999)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_evidence(conn, ids["edge"], 999)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_mention(conn, 999, ids["source"])
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_alias(conn, "Nobody", 999)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        add_source(conn, 999, url="https://example.com/orphan")
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        conn.execute("DELETE FROM nodes WHERE id = ?", (ids["alice"],))
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_uniqueness_constraints(graph):
    conn, ids = graph
    duplicates = [
        lambda: add_node(conn, "  ALICE   example "),
        lambda: add_edge(conn, ids["alice"], ids["acme"], "WORKS_FOR"),
        lambda: add_evidence(conn, ids["edge"], ids["source"], 3, 4),
        lambda: add_mention(conn, ids["alice"], ids["source"], 2),
        lambda: add_source(conn, ids["run"]),
    ]
    for insert in duplicates:
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            insert()
    add_alias(conn, "AE", ids["alice"])
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        add_alias(conn, "ae", ids["alice"])

    # Allowed: same name with another type, another relation type, another source version, an ambiguous alias.
    add_node(conn, "Alice Example", "ORG")
    add_edge(conn, ids["alice"], ids["acme"], "CRITICIZED")
    add_source(conn, ids["run"], content_hash="hash-a-edited")
    add_alias(conn, "AE", ids["acme"])
    assert conn.execute("SELECT count(*) FROM node_aliases WHERE alias_key = 'ae'").fetchone()[0] == 2


@pytest.mark.parametrize(("sql", "params"), [
    ("UPDATE nodes SET entity_type = 'EVENT', key = 'EVENT:alice example' WHERE id = ?", "alice"),
    ("UPDATE nodes SET key = 'PERSON:someone else' WHERE id = ?", "alice"),
    ("UPDATE nodes SET mention_count = -1 WHERE id = ?", "alice"),
    ("UPDATE nodes SET first_seen = '2026-10-07 12:00:00' WHERE id = ?", "alice"),
    ("UPDATE edges SET target_node_id = source_node_id WHERE id = ?", "edge"),
    ("UPDATE edges SET weight = 0 WHERE id = ?", "edge"),
    ("UPDATE edges SET directed = 2 WHERE id = ?", "edge"),
    ("UPDATE edges SET directed = 0, source_node_id = target_node_id, target_node_id = source_node_id WHERE id = ?",
     "edge"),
    ("UPDATE edges SET last_seen = '2026-10-06T00:00:00.000000Z' WHERE id = ?", "edge"),
    ("UPDATE edges SET weight = 'many' WHERE id = ?", "edge"),
    ("UPDATE edge_sources SET start_offset = -1 WHERE edge_id = ?", "edge"),
    ("UPDATE edge_sources SET end_offset = 0, start_offset = 5 WHERE edge_id = ?", "edge"),
    ("UPDATE edge_sources SET quality_tier = 'guess' WHERE edge_id = ?", "edge"),
    ("UPDATE node_mentions SET occurrence_count = 0 WHERE node_id = ?", "alice"),
    ("UPDATE node_mentions SET surface_forms_json = 'not json' WHERE node_id = ?", "alice"),
    ("UPDATE sources SET source_type = 'blog' WHERE id = ?", "source"),
    ("UPDATE sources SET last_scraped_at = '2026-10-06T00:00:00.000000Z' WHERE id = ?", "source"),
    ("UPDATE sources SET published_at = '2026-10-07' WHERE id = ?", "source"),
    ("UPDATE crawl_runs SET status = 'done', finished_at = started_at WHERE id = ?", "run"),
    ("UPDATE crawl_runs SET status = 'complete' WHERE id = ?", "run"),
    ("UPDATE crawl_runs SET finished_at = '2026-10-06T00:00:00.000000Z' WHERE id = ?", "run"),
])
def test_check_constraints(graph, sql, params):
    conn, ids = graph
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, (ids[params],))


def test_valid_symmetric_edge_and_finished_run(graph):
    conn, ids = graph
    low, high = sorted((ids["alice"], ids["acme"]))
    add_edge(conn, low, high, "MET_WITH", directed=0)
    with pytest.raises(sqlite3.IntegrityError):
        add_edge(conn, high, low, "ALLIED_WITH", directed=0)
    conn.execute("UPDATE crawl_runs SET status = 'partial', finished_at = ? WHERE id = ?", (T1, ids["run"]))
    conn.execute("UPDATE sources SET published_at = ?, last_scraped_at = ? WHERE id = ?", (T0, T1, ids["source"]))


def test_transaction_rolls_back_on_exception(graph):
    conn, ids = graph
    with pytest.raises(RuntimeError, match="mid-item"), db.transaction(conn):
        source = add_source(conn, ids["run"], url="https://example.com/b", content_hash="hash-b")
        bob = add_node(conn, "Bob Example")
        edge = add_edge(conn, bob, ids["acme"])
        add_evidence(conn, edge, source)
        raise RuntimeError("mid-item failure")
    assert not conn.in_transaction
    counts = [conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
              for table in ("sources", "nodes", "edges", "edge_sources")]
    assert counts == [1, 2, 1, 1]


def test_transaction_rolls_back_on_constraint_error_and_commits_otherwise(graph):
    conn, ids = graph
    with pytest.raises(sqlite3.IntegrityError), db.transaction(conn):
        add_node(conn, "Bob Example")
        add_edge(conn, ids["alice"], ids["alice"])
    assert conn.execute("SELECT count(*) FROM nodes WHERE name = 'Bob Example'").fetchone()[0] == 0
    with db.transaction(conn):
        add_node(conn, "Bob Example")
        with pytest.raises(sqlite3.OperationalError), db.transaction(conn):
            pass
    assert conn.execute("SELECT count(*) FROM nodes WHERE name = 'Bob Example'").fetchone()[0] == 1


def test_writer_lock_is_visible_to_second_connection(tmp_path, graph):
    conn, _ = graph
    other = db.connect(tmp_path / "graph.sqlite3", busy_timeout_ms=50)
    with db.transaction(conn):
        add_node(conn, "Carol Example")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.execute("BEGIN IMMEDIATE")
        assert other.execute("SELECT count(*) FROM nodes").fetchone()[0] == 2
    assert other.execute("SELECT count(*) FROM nodes").fetchone()[0] == 3
    other.close()


def test_integrity_report(graph):
    conn, ids = graph
    report = db.integrity_report(conn)
    assert set(report) >= {"foreign_key_violations", "edges_without_evidence", "edge_weight_mismatches"}
    assert not any(report.values())

    # A second version of the same URL adds evidence but must not add weight.
    edited = add_source(conn, ids["run"], content_hash="hash-a-edited")
    add_evidence(conn, ids["edge"], edited)
    assert not any(db.integrity_report(conn).values())

    conn.execute("UPDATE edges SET weight = 2 WHERE id = ?", (ids["edge"],))
    conn.execute("UPDATE nodes SET mention_count = 3, first_seen = ? WHERE id = ?", (T1, ids["alice"]))
    add_edge(conn, ids["acme"], ids["alice"], "SUED")
    report = db.integrity_report(conn)
    assert report["edge_weight_mismatches"] == 2
    assert report["edges_without_evidence"] == 1
    assert report["node_mention_count_mismatches"] == 1
    assert report["node_first_seen_mismatches"] == 1
    assert report["edge_first_seen_mismatches"] == 0
