"""Offline ingest pipeline tests: a fake crawler replaces Crawl4AI at the network boundary.

Every HTML page, URL, handle, and sentence below is labelled synthetic: written for these tests in the shape of a
news article, a Reddit thread (shreddit-post/shreddit-comment), and an X post (article[data-testid=tweet]). Nothing
here is crawled evidence, and the events described are not claims about the real world. These runs exercise the
real normalization, spaCy extraction, SQLite storage, and API code; they are not a live crawl.
"""

import asyncio
import json
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from media_intelligence import api, db, pipeline, storage
from media_intelligence.config import Settings, load_config
from media_intelligence.extraction import Extractor
from media_intelligence.models import Extraction, FetchResult, timestamp
from media_intelligence.pipeline import ingest, safe_url
from media_intelligence.storage import verify_integrity

CONFIG = Path(__file__).resolve().parents[2] / "config"
T1 = datetime(2026, 1, 5, 12, tzinfo=UTC)
T2 = T1 + timedelta(days=1)
EPOCH = datetime(2000, 1, 1, tzinfo=UTC)

NEWS_URL = "https://news.example.com/world/synthetic-talks"
REDDIT_URL = "https://www.reddit.com/r/geopolitics/comments/abc123/synthetic_thread/"
X_URL = "https://x.com/SyntheticDesk/status/1234567890"

SOURCES_TOML = f"""# Synthetic offline test configuration; these URLs are never fetched.
aliases_file = "aliases.toml"
topics_file = "topics.toml"

[crawl]
max_depth = 0
max_pages_total = 10

[[sources]]
name = "news"
source_type = "news"
adapter = "article"
allowed_domains = ["news.example.com"]
seeds = ["{NEWS_URL}"]

[[sources]]
name = "reddit"
source_type = "discussion"
adapter = "reddit"
allowed_domains = ["www.reddit.com"]
include_paths = ["^/r/geopolitics/comments/"]
seeds = ["{REDDIT_URL}"]

[[sources]]
name = "x"
source_type = "microblog"
adapter = "x"
allowed_domains = ["x.com"]
include_paths = ["^/[^/]+/status/[0-9]+"]
seeds = ["{X_URL}"]
"""

# Synthetic news article.
NEWS_HTML = """<html><head><title>Synthetic talks report</title>
<script type="application/ld+json">{"@type": "NewsArticle", "headline": "Synthetic: envoys meet in Jerusalem",
 "author": {"name": "Synthetic Reporter"}, "datePublished": "2026-01-04T09:00:00Z"}</script></head>
<body><nav>Home World Sport</nav><article>
<p>Antony Blinken met Benjamin Netanyahu in Jerusalem on Monday.</p>
<p>Mark Rutte, the head of NATO, criticized Hungary over its veto.</p>
<p>The two delegations discussed the ceasefire and sanctions.</p>
</article><footer>Synthetic footer</footer></body></html>"""

# Synthetic Reddit thread: a post, a comment, and a nested reply.
REDDIT_HTML = """<html><head><title>Synthetic thread : r/geopolitics</title></head><body>
<shreddit-post id="t3_abc123" post-title="Synthetic: who met whom this week?" author="synthetic_poster"
  created-timestamp="2026-01-04T08:00:00+00:00">
  <div slot="text-body"><p>Joe Biden met with Xi Jinping on the sidelines of the summit.</p></div>
</shreddit-post>
<shreddit-comment thingid="t1_c1" author="synthetic_alice" parentid="t3_abc123" created="2026-01-04T09:00:00Z">
  <div slot="comment"><p>Antony Blinken met Benjamin Netanyahu again, which is the real story.</p></div>
  <shreddit-comment thingid="t1_c2" author="synthetic_bob" parentid="t1_c1">
    <div slot="comment"><p>Emmanuel Macron and Olaf Scholz met in Berlin on Friday.</p></div>
  </shreddit-comment>
</shreddit-comment></body></html>"""

# Synthetic X post page with an unrelated post before the target.
X_HTML = """<html><body>
<article data-testid="tweet"><a href="/Other/status/42">Other</a>
  <div data-testid="tweetText">An unrelated synthetic post.</div></article>
<article data-testid="tweet"><a href="/SyntheticDesk/status/1234567890">Post</a>
  <div data-testid="User-Name">Synthetic Desk @SyntheticDesk</div>
  <div data-testid="tweetText">Antony Blinken met Benjamin Netanyahu in Jerusalem today.</div>
  <time datetime="2026-01-05T10:00:00.000Z"></time></article></body></html>"""

# Synthetic X login wall: the shell X serves instead of the post when a session is required.
X_LOGIN_WALL = """<html><head><title>X</title></head><body><div>Log in to X</div>
<div>Sign up now to see what is happening.</div></body></html>"""


class FakeCrawler:
    """Stands in for BrowserCrawler at the network boundary and yields prepared pages."""

    def __init__(self, pages):
        self._pages = pages

    async def pages(self):
        for page in self._pages:
            yield page


class StubExtractor:
    """No-NLP extractor for failure-path tests; it emits no entities and can fail for chosen sources."""

    version = "stub-1"

    def __init__(self, fail_for=()):
        self.fail_for = set(fail_for)

    def extract(self, item):
        if item.metadata["source_name"] in self.fail_for:
            raise RuntimeError("synthetic NLP failure")
        return Extraction()


class CountingExtractor:
    """Wraps the real extractor and counts calls, to prove unchanged items skip spaCy."""

    def __init__(self, inner):
        self.inner, self.version, self.calls = inner, inner.version, 0

    def extract(self, item):
        self.calls += 1
        return self.inner.extract(item)


def page(source, url, html="", at=T1, *, success=True, status=200, error=None, depth=0):
    return FetchResult(requested_url=url, final_url=url, source_name=source, depth=depth, fetched_at=at,
                       success=success, status_code=status, html=html, error=error)


def good_pages(at=T1, x_html=X_HTML):
    return [page("news", NEWS_URL, NEWS_HTML, at), page("reddit", REDDIT_URL, REDDIT_HTML, at),
            page("x", X_URL, x_html, at)]


def factory(pages):
    return lambda config, settings: FakeCrawler(pages)


@pytest.fixture(scope="module")
def extractor():
    return Extractor("en_core_web_sm", CONFIG / "aliases.toml", CONFIG / "topics.toml")


@pytest.fixture(autouse=True)
def store_clock(monkeypatch):
    """Pin the store-time clock before every synthetic fetch time, so observations equal the fetch times."""
    monkeypatch.setattr(storage, "utcnow", lambda: EPOCH)


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "config"
    folder.mkdir()
    shutil.copy(CONFIG / "aliases.toml", folder)
    shutil.copy(CONFIG / "topics.toml", folder)
    path = folder / "sources.toml"
    path.write_text(SOURCES_TOML)
    settings = Settings(database_path=tmp_path / "data" / "graph.sqlite3", report_dir=tmp_path / "reports")
    return SimpleNamespace(config=load_config(path), settings=settings, path=path, folder=folder)


def run(env, pages, extractor, capsys):
    code = asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=factory(pages),
                              extractor=extractor))
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(Path(summary["report"]).read_text())
    assert summary["exit_code"] == code == report["exit_code"]
    for tally in report["sources"].values():
        assert tally["attempted"] == sum(tally[name] for name in pipeline.OUTCOMES)
    return code, report


def reader(env):
    return closing(db.connect(env.settings.database_path, readonly=True))


def snapshot(conn):
    return {"sources": conn.execute("SELECT count(*) FROM sources").fetchone()[0],
            "evidence": conn.execute("SELECT count(*) FROM edge_sources").fetchone()[0],
            "edges": conn.execute("SELECT id, weight, first_seen FROM edges ORDER BY id").fetchall(),
            "nodes": conn.execute("SELECT id, mention_count, first_seen FROM nodes ORDER BY id").fetchall()}


def met_edge(conn):
    return conn.execute(
        """SELECT e.* FROM edges e JOIN nodes a ON a.id = e.source_node_id JOIN nodes b ON b.id = e.target_node_id
           WHERE e.relation_type = 'met_with'
             AND 'PERSON:antony blinken' IN (a.key, b.key) AND 'PERSON:benjamin netanyahu' IN (a.key, b.key)"""
    ).fetchone()


def test_end_to_end_ingest_stores_three_sources_with_semantic_evidence(env, extractor, capsys):
    code, report = run(env, good_pages(), extractor, capsys)
    assert code == 0 and report["status"] == "complete" and report["live"] is False
    assert report["missing_sources"] == [] and report["integrity"] == {"ok": True, "problems": []}
    assert {name: s["stored_new"] for name, s in report["sources"].items()} == {"news": 1, "reddit": 1, "x": 1}
    assert report["edge_quality"]["semantic_rule"] >= 4 and report["edge_quality"]["cooccurrence"] >= 1
    assert report["extractor_version"] == extractor.version
    hashed = b"\0".join((env.folder / name).read_bytes() for name in ("sources.toml", "aliases.toml", "topics.toml"))
    assert report["config_hash"] == sha256(hashed).hexdigest()
    assert report["sources"]["reddit"]["segments"] == 3 and report["sources"]["x"]["missing_metadata"]["title"] == 1
    assert report["storage"]["invalid_evidence"] == 0 and report["warnings"] == []
    assert set(report["graph"]["nodes_by_type"]) == {"LOCATION", "ORG", "PERSON", "TOPIC"}
    assert report["graph"]["nodes_by_type"]["PERSON"] == 7  # Blinken, Netanyahu, Rutte, Biden, Xi, Macron, Scholz

    with reader(env) as conn:
        assert verify_integrity(conn) == []
        assert {r[0] for r in conn.execute("SELECT source_type FROM sources")} == {"news", "discussion", "microblog"}
        run_row = conn.execute("SELECT * FROM crawl_runs").fetchone()
        assert run_row["status"] == "complete" and run_row["extractor_version"] == extractor.version
        assert json.loads(run_row["summary_json"])["report"] == str(Path(report["report"]))
        edge = met_edge(conn)
        assert edge["weight"] == 3 and edge["directed"] == 0 and edge["first_seen"] == timestamp(T1)
        evidence = conn.execute("""SELECT es.*, s.segments_json FROM edge_sources es JOIN sources s
                                   ON s.id = es.source_id WHERE es.edge_id = ?""", (edge["id"],)).fetchall()
        assert len(evidence) == 3 and {e["quality_tier"] for e in evidence} == {"semantic_rule"}
        for row in evidence:
            text = {s["id"]: s["text"] for s in json.loads(row["segments_json"])}[row["segment_id"]]
            assert text[row["start_offset"]:row["end_offset"]] == row["evidence_text"]
        assert conn.execute("SELECT mention_count FROM nodes WHERE key = 'PERSON:antony blinken'").fetchone()[0] == 3
        criticized = conn.execute("""SELECT count(*) FROM edges e JOIN nodes a ON a.id = e.source_node_id
                                     JOIN nodes b ON b.id = e.target_node_id WHERE e.relation_type = 'criticized'
                                     AND a.key = 'PERSON:mark rutte' AND b.key = 'LOCATION:hungary'""").fetchone()
        assert criticized[0] == 1


def test_identical_rerun_adds_no_support_and_skips_extraction(env, extractor, capsys):
    counting = CountingExtractor(extractor)
    assert run(env, good_pages(T1), counting, capsys)[0] == 0
    assert counting.calls == 3
    with reader(env) as conn:
        before = snapshot(conn)
    code, report = run(env, good_pages(T2), counting, capsys)
    assert code == 0 and counting.calls == 3
    assert {name: (s["stored_new"], s["stored_unchanged"]) for name, s in report["sources"].items()} == {
        "news": (0, 1), "reddit": (0, 1), "x": (0, 1)}
    assert report["storage"]["evidence_added"] == 0 and report["integrity"]["ok"]
    with reader(env) as conn:
        assert snapshot(conn) == before
        assert {r[0] for r in conn.execute("SELECT last_scraped_at FROM sources")} == {timestamp(T2)}
        assert met_edge(conn)["last_seen"] == timestamp(T2)
        assert [r[0] for r in conn.execute("SELECT status FROM crawl_runs ORDER BY id")] == ["complete", "complete"]


def test_login_wall_and_failed_pages_make_a_partial_run(env, extractor, capsys, caplog):
    caplog.set_level("INFO")
    pages = [page("news", NEWS_URL, NEWS_HTML),
             page("news", "https://news.example.com/world/slow?token=secret", success=False, status=None,
                  error="TimeoutError: navigation exceeded 30s at https://news.example.com/world/slow?token=secret"),
             page("news", "https://news.example.com/world/empty", "<article><p>Too short.</p></article>"),
             page("reddit", REDDIT_URL, REDDIT_HTML),
             page("reddit", "https://www.reddit.com/r/geopolitics/comments/zzz/", success=False, status=403),
             page("x", X_URL, X_LOGIN_WALL)]
    code, report = run(env, pages, extractor, capsys)
    assert code == 3 and report["status"] == "partial" and report["missing_sources"] == ["x"]
    news, reddit, x = (report["sources"][name] for name in ("news", "reddit", "x"))
    assert (news["attempted"], news["fetched"], news["failed"], news["skipped"], news["stored_new"]) == (3, 2, 1, 1, 1)
    assert news["reasons"] == {"failed": {"TimeoutError": 1}, "skipped": {"empty_content": 1}}
    assert reddit["reasons"] == {"blocked": {"http_403": 1}} and reddit["stored_new"] == 1
    assert (x["attempted"], x["fetched"], x["blocked"], x["stored"]) == (1, 1, 1, 0)
    assert x["reasons"] == {"blocked": {"access_or_structure": 1}}
    assert report["skip_reasons"]["blocked"] == {"access_or_structure": 1, "http_403": 1}
    assert any("x" in warning and "Required sources" in warning for warning in report["warnings"])
    assert "secret" not in json.dumps(report) and "secret" not in caplog.text
    assert "Antony Blinken met" not in caplog.text  # page bodies never reach the log
    with reader(env) as conn:
        assert conn.execute("SELECT status FROM crawl_runs").fetchone()[0] == "partial"
        assert conn.execute("SELECT count(*) FROM sources").fetchone()[0] == 2


def test_api_answers_over_ingested_graph(env, extractor, capsys):
    assert run(env, good_pages(), extractor, capsys)[0] == 0
    client = TestClient(api.create_app(Settings(database_path=env.settings.database_path)))

    network = client.get("/entity/Antony Blinken/network", params={"depth": 1})
    assert network.status_code == 200
    body = network.json()
    names = {node["id"]: node["name"] for node in body["nodes"]}
    assert names[body["root_id"]] == "Antony Blinken" and "Benjamin Netanyahu" in names.values()
    met = [e for e in body["edges"] if e["relation_type"] == "met_with"
           and {names[e["source"]], names[e["target"]]} == {"Antony Blinken", "Benjamin Netanyahu"}]
    assert len(met) == 1 and met[0]["weight"] == 3 and met[0]["directed"] is False
    two_hop = client.get("/entity/antony blinken/network", params={"depth": 2}).json()
    assert {n["id"] for n in body["nodes"]} <= {n["id"] for n in two_hop["nodes"]}

    connections = client.get("/connections/new", params={"since": "2026-01-01T00:00:00Z"})
    assert connections.status_code == 200
    items = {(i["source"]["name"], i["relation_type"], i["target"]["name"]): i for i in connections.json()["items"]}
    pair = next(i for key, i in items.items() if key[1] == "met_with" and "Antony Blinken" in key)
    assert pair["reason"] == "new" and pair["weight_now"] == 3
    later = client.get("/connections/new", params={"since": timestamp(T2)}).json()
    assert later["items"] == [] and later["total"] == 0

    central = client.get("/entities/central")
    assert central.status_code == 200
    ranked = central.json()["items"]
    assert ranked and ranked[0]["rank"] == 1 and ranked[0]["degree"] >= 1
    assert "Antony Blinken" in {item["name"] for item in ranked}

    # Reviewed aliases and handles resolve even though the synthetic pages only ever write the full names.
    for name in ("Blinken", "EmmanuelMacron", "@EmmanuelMacron", "Zelensky"):
        expected = 404 if name == "Zelensky" else 200  # Zelenskyy's node does not exist in this graph
        assert client.get(f"/entity/{name}/network", params={"depth": 1}).status_code == expected, name

    evidence = client.get(f"/edges/{met[0]['id']}/sources").json()
    assert evidence["total"] == 3 and all("met" in item["evidence_text"] for item in evidence["items"])
    assert {item["source_type"] for item in evidence["items"]} == {"news", "discussion", "microblog"}


def test_persistent_database_failure_aborts_with_failed_status(env, capsys, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(pipeline, "store_item", broken)
    code, report = run(env, good_pages(), StubExtractor(), capsys)
    assert code == 1 and report["status"] == "failed"
    assert report["fatal_error"] == "database_error: OperationalError"
    assert report["totals"]["attempted"] == 2 and report["totals"]["storage_failed"] == 2  # the X page never ran
    with reader(env) as conn:
        assert conn.execute("SELECT status FROM crawl_runs").fetchone()[0] == "failed"


def test_item_failures_are_counted_and_do_not_abort(env, capsys, monkeypatch):
    calls = []

    def flaky(*args, **kwargs):
        # Synthetic: the first and third writes hit a transient error; the success between resets the streak.
        calls.append(1)
        if len(calls) in (1, 3):
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    original = pipeline.store_item
    monkeypatch.setattr(pipeline, "store_item", flaky)
    code, report = run(env, good_pages(), StubExtractor(), capsys)
    assert code == 3 and report["missing_sources"] == ["news", "x"] and report["fatal_error"] is None
    assert report["sources"]["news"]["reasons"] == {"storage_failed": {"database_error: OperationalError": 1}}
    assert "The graph has no edges" in report["warnings"]  # the stub extractor emits nothing

    # The unchanged reddit version is refreshed without calling the (here failing) extractor.
    monkeypatch.setattr(pipeline, "store_item", original)
    code, report = run(env, good_pages(T2), StubExtractor(fail_for={"reddit"}), capsys)
    assert code == 0 and report["status"] == "complete" and report["missing_sources"] == []
    assert report["sources"]["reddit"]["stored_unchanged"] == 1
    assert report["sources"]["news"]["stored_new"] == report["sources"]["x"]["stored_new"] == 1


def test_extraction_failure_skips_the_item(env, capsys):
    code, report = run(env, good_pages(), StubExtractor(fail_for={"reddit"}), capsys)
    assert code == 3 and report["missing_sources"] == ["reddit"]
    assert report["sources"]["reddit"]["reasons"] == {"extraction_failed": {"extraction_error: RuntimeError": 1}}
    with reader(env) as conn:
        assert conn.execute("SELECT count(*) FROM sources WHERE source_type = 'discussion'").fetchone()[0] == 0


@pytest.mark.parametrize("error", [RuntimeError("synthetic browser crash"), OSError("synthetic socket error"),
                                   ValueError("synthetic bad value")])
def test_crawler_crash_records_a_failed_run_and_returns_its_exit_code(env, capsys, error):
    class Crashing(FakeCrawler):
        async def pages(self):
            yield page("news", NEWS_URL, NEWS_HTML)
            raise error

    code = asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=lambda c, s: Crashing([]),
                              extractor=StubExtractor()))
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(Path(summary["report"]).read_text())
    # The process exit code and the recorded one agree (the CLI maps a re-raised OSError/ValueError to 2).
    assert code == summary["exit_code"] == report["exit_code"] == 1
    assert report["status"] == "failed" and report["fatal_error"].startswith(f"crawl_aborted: {type(error).__name__}")
    assert report["sources"]["news"]["stored_new"] == 1  # committed items stay committed
    with reader(env) as conn:
        assert conn.execute("SELECT status FROM crawl_runs").fetchone()[0] == "failed"


def test_interrupt_is_recorded_and_propagates(env, capsys):
    class Interrupted(FakeCrawler):
        async def pages(self):
            yield page("news", NEWS_URL, NEWS_HTML)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=lambda c, s: Interrupted([]),
                           extractor=StubExtractor()))
    with reader(env) as conn:
        row = conn.execute("SELECT status, finished_at FROM crawl_runs").fetchone()
        assert row["status"] == "failed" and row["finished_at"] is not None


def test_crawler_setup_error_is_raised_before_a_run_is_recorded(env):
    def broken(config, settings):
        raise ValueError("MI_BROWSER_STATE_PATH must point to an existing local storage-state JSON file")

    with pytest.raises(ValueError, match="BROWSER_STATE"):
        asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=broken, extractor=StubExtractor()))
    with reader(env) as conn:
        assert conn.execute("SELECT count(*) FROM crawl_runs").fetchone()[0] == 0
    settings = env.settings.model_copy(update={"browser_state_path": env.folder / "synthetic-missing-state.json"})
    with pytest.raises(ValueError, match="BROWSER_STATE"):
        pipeline.BrowserCrawler(env.config, settings)


def test_setup_errors_fail_before_touching_the_database(env):
    (env.folder / "topics.toml").unlink()
    with pytest.raises(ValueError, match="topics.toml"):
        asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=factory([])))
    shutil.copy(CONFIG / "topics.toml", env.folder)
    settings = env.settings.model_copy(update={"spacy_model": "xx_synthetic_missing_model"})
    with pytest.raises(ValueError, match="spaCy model"):
        asyncio.run(ingest(env.config, settings, env.path, crawler_factory=factory([])))
    assert not env.settings.database_path.exists()


def test_safe_url_drops_credentials_query_and_fragment():
    assert safe_url("https://user:pw@x.com:8443/a/b?token=1#frag") == "https://x.com:8443/a/b"
    assert safe_url("https://[::1") == "<invalid url>"


def test_failed_seed_makes_the_run_partial_but_a_failed_followed_link_does_not(env, extractor, capsys):
    extra_seed = page("news", "https://news.example.com/world/blocked-seed", success=False, status=403)
    code, report = run(env, good_pages() + [extra_seed], extractor, capsys)
    assert code == 3 and report["status"] == "partial" and report["missing_sources"] == []
    assert report["failed_seeds"] == {"news": 1}
    assert any("Seed pages without a stored item" in w and "news=1" in w for w in report["warnings"])

    link = page("news", "https://news.example.com/world/followed", success=False, status=404, depth=1, at=T2)
    code, report = run(env, good_pages(T2) + [link], extractor, capsys)
    assert code == 0 and report["failed_seeds"] == {} and report["sources"]["news"]["failed"] == 1


def test_configured_listing_seed_is_not_a_failed_seed(env, extractor, capsys):
    text = env.path.read_text().replace('seeds = ["https://news.example.com/world/synthetic-talks"]',
                                        'seeds = ["https://news.example.com/world/synthetic-talks"]\n'
                                        'listing_paths = ["^/world$"]')
    env.path.write_text(text)
    env.config = load_config(env.path)
    listing = page("news", "https://news.example.com/world", "<main>" + "Synthetic headline link. " * 20 + "</main>")
    code, report = run(env, [listing] + good_pages(), extractor, capsys)
    assert code == 0 and report["failed_seeds"] == {}
    assert report["sources"]["news"]["reasons"] == {"skipped": {"listing_page": 1}}


def test_rerun_with_changed_extraction_rules_is_refused(env, capsys):
    assert run(env, good_pages(), StubExtractor(), capsys)[0] == 0
    changed = StubExtractor()
    changed.version = "stub-2"  # synthetic: e.g. aliases.toml was edited between runs
    with pytest.raises(ValueError, match="explicit rebuild"):
        asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=factory(good_pages(T2)),
                           extractor=changed))
    with reader(env) as conn:
        assert [tuple(r) for r in conn.execute("SELECT extractor_version, status FROM crawl_runs")] == [
            ("stub-1", "complete")]


def test_unwritable_report_folder_still_records_the_run(env, capsys):
    env.settings.report_dir.mkdir(parents=True)
    env.settings.report_dir.chmod(0o555)
    try:
        code = asyncio.run(ingest(env.config, env.settings, env.path, crawler_factory=factory(good_pages()),
                                  extractor=StubExtractor()))
    finally:
        env.settings.report_dir.chmod(0o755)
    summary = json.loads(capsys.readouterr().out)
    assert code == summary["exit_code"] == 0 and summary["report"] is None
    assert any("Run report could not be written" in w for w in summary["warnings"])
    with reader(env) as conn:
        row = conn.execute("SELECT status, finished_at, summary_json FROM crawl_runs").fetchone()
        assert row["status"] == "complete" and row["finished_at"] is not None
        assert json.loads(row["summary_json"])["report"] is None


def test_separated_transient_database_errors_do_not_abort_the_run(env, capsys, monkeypatch):
    outcomes = iter([sqlite3.OperationalError("database is locked"), ValueError("synthetic rejected item"),
                     sqlite3.OperationalError("database is locked")])

    def flaky(*args, **kwargs):
        if (error := next(outcomes, None)) is not None:
            raise error
        return original(*args, **kwargs)

    original = pipeline.store_item
    monkeypatch.setattr(pipeline, "store_item", flaky)
    pages = [page("news", f"https://news.example.com/world/story-{n}", NEWS_HTML) for n in "abcd"]
    code, report = run(env, pages, StubExtractor(), capsys)
    assert report["fatal_error"] is None and report["status"] != "failed" and code != 1
    assert report["totals"]["attempted"] == 4 and report["totals"]["stored_new"] == 1
    assert report["sources"]["news"]["reasons"]["storage_failed"] == {
        "database_error: OperationalError": 2, "storage_rejected: ValueError": 1}


def test_observation_time_is_the_store_time_so_polling_cannot_miss_it(env, extractor, capsys, monkeypatch):
    stored_at = T1 + timedelta(hours=1)  # synthetic: extraction and the write lock took an hour after the fetch
    monkeypatch.setattr(storage, "utcnow", lambda: stored_at)
    assert run(env, good_pages(T1), extractor, capsys)[0] == 0
    with reader(env) as conn:
        assert {r[0] for r in conn.execute("SELECT scraped_at FROM sources")} == {timestamp(T1)}
        assert {r[0] for r in conn.execute("SELECT observed_at FROM edge_sources")} == {timestamp(stored_at)}
        assert verify_integrity(conn) == []
    # A client whose previous as_of fell between the fetch and the commit still sees the edges as new.
    client = TestClient(api.create_app(Settings(database_path=env.settings.database_path)))
    later = client.get("/connections/new", params={"since": timestamp(T1 + timedelta(minutes=30))}).json()
    assert later["total"] > 0 and {item["reason"] for item in later["items"]} == {"new"}


def test_thread_permalink_variants_count_as_one_logical_source(env, extractor, capsys):
    # Synthetic comment-permalink view of the same thread: only the post and one comment are shown.
    permalink_html = REDDIT_HTML.split("<shreddit-comment thingid=\"t1_c2\"")[0] + "</shreddit-comment></body></html>"
    permalink = page("reddit", "https://www.reddit.com/r/geopolitics/comments/abc123/comment/c1/?sort=new",
                     permalink_html)
    code, report = run(env, good_pages() + [permalink], extractor, capsys)
    assert code == 0 and report["sources"]["reddit"]["stored_new"] == 2  # two versions of one thread
    with reader(env) as conn:
        urls = [r[0] for r in conn.execute("SELECT source_url FROM sources WHERE source_type = 'discussion'")]
        assert urls == [REDDIT_URL.replace("synthetic_thread/", "")] * 2
        assert met_edge(conn)["weight"] == 3  # news, the thread, and the X post; the permalink adds nothing
        assert verify_integrity(conn) == []


# Synthetic pages in the shapes of the three shipped adapters (Al Jazeera article body, Hacker News thread,
# Mastodon thread). Nothing here is crawled evidence.
SHIPPED_ARTICLE = """<html><head><script type="application/ld+json">{"@type": "NewsArticle",
 "headline": "Synthetic: envoys meet in Jerusalem", "datePublished": "2026-01-04T09:00:00Z"}</script></head><body>
<div class="wysiwyg"><p>Antony Blinken met Benjamin Netanyahu in Jerusalem on Monday.</p>
<section class="more-on"><h2>Recommended Stories</h2><ul><li>list 1 of 1Emmanuel Macron criticized NATO</li></ul></section>
<div class="sib-newsletter-form"><h4>Sign up for Al Jazeera</h4><span>protected by reCAPTCHA</span></div>
<p>The two delegations discussed the ceasefire and sanctions at length, officials said.</p></div></body></html>"""
SHIPPED_HN = """<html><table>
<tr class="athing submission" id="49458161"><td><span class="titleline"><a href="https://news.example/hf">
Nvidia agrees to acquire Hugging Face for $13B</a></span></td></tr>
<tr><td class="subtext"><span class="subline"><a class="hnuser" href="user?id=poster">poster</a>
<span class="age" title="2026-03-12T10:00:00 1773309600"><a href="item?id=49458161">1 day ago</a></span></span></td></tr>
<tr class="athing comtr" id="49458200"><td><td class="ind" indent="0"></td><span class="comhead">
<a class="hnuser">alice</a><span class="age" title="2026-03-12T11:00:00 1773313200"></span></span>
<div class="comment"><div class="commtext c00">Nvidia and AMD both want the open model ecosystem.</div></div></td></tr>
</table></html>"""
SHIPPED_MASTODON = """<html><head><meta property="og:published_time" content="2026-09-30T10:00:00Z"></head><body>
<div class="detailed-status"><span class="display-name__account">@arstechnica@mastodon.social</span>
<div class="status__content__text"><p>Google announces Gemini 4 Argon AI model, but you can't use it yet</p></div></div>
<div class="status status-reply" data-id="101"><span class="display-name__account">@reader@mastodon.social</span>
<div class="status__content__text"><p>Microsoft and Google will fight over this.</p></div></div>
</body></html>"""


def test_shipped_config_and_adapters_ingest_offline_with_typed_edges_from_every_source_type(tmp_path, extractor,
                                                                                            capsys):
    """The shipped sources.toml (unchanged) with its three adapters, end to end through ingest and the run report."""
    folder = tmp_path / "config"
    folder.mkdir()
    for name in ("sources.toml", "aliases.toml", "topics.toml"):
        shutil.copy(CONFIG / name, folder)
    config = load_config(folder / "sources.toml")
    seeds = {s.name: s.seeds[0] for s in config.sources}
    pages = [page("aljazeera", seeds["aljazeera"], SHIPPED_ARTICLE), page("hackernews", seeds["hackernews"], SHIPPED_HN),
             page("mastodon", seeds["mastodon"], SHIPPED_MASTODON)]
    shipped = SimpleNamespace(config=config, path=folder / "sources.toml", folder=folder,
                              settings=Settings(database_path=tmp_path / "graph.sqlite3", report_dir=tmp_path / "r"))
    code, report = run(shipped, pages, extractor, capsys)
    assert code == 0 and report["integrity"]["ok"]
    assert {name: s["stored_new"] for name, s in report["sources"].items()} == {
        "aljazeera": 1, "hackernews": 1, "mastodon": 1}
    by_type = report["graph"]["semantic_evidence_by_source_type_and_relation"]
    assert by_type == {"news": {"met_with": 1}, "discussion": {"acquired": 1},
                       "microblog": {"discussed_topic": 1, "released": 1}}
    assert set(report["graph"]["evidence_by_source_type_and_tier"]) == {"news", "discussion", "microblog"}
    with reader(shipped) as conn:
        body = conn.execute("SELECT body FROM sources WHERE source_type = 'news'").fetchone()[0]
        assert "Recommended Stories" not in body and "reCAPTCHA" not in body
        assert conn.execute("SELECT count(*) FROM nodes WHERE key = 'PERSON:emmanuel macron'").fetchone()[0] == 0


def test_crawl4ai_failure_reason_keeps_the_cause_line():
    from media_intelligence.pipeline import _code
    message = ("Unexpected error in _crawl_web at line 778 in _crawl_web (.venv/lib/crawl4ai/x.py):\n"
               "Error: Failed on navigating ACS-GOTO:\nPage.goto: Timeout 30000ms exceeded at https://example.com/a")
    assert _code(message) == "crawl_failed: Page.goto: Timeout 30000ms exceeded at <url>"
    assert _code("empty_content: no substantial article body") == "empty_content"
