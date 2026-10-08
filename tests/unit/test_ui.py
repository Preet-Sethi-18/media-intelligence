"""Browser interface: page served, overview from a real (empty) schema, pipeline runner without crawling."""

import sys

from fastapi.testclient import TestClient

from media_intelligence import db
from media_intelligence.api import create_app
from media_intelligence.config import Settings


def client(tmp_path, *, migrate=True):
    path = tmp_path / "graph.sqlite3"
    if migrate:
        conn = db.connect(path)
        db.migrate(conn)
        conn.close()
    settings = Settings(database_path=path, report_dir=tmp_path / "reports", config_path="config/sources.toml")
    return TestClient(create_app(settings))


def test_page_and_assets_are_served_and_root_redirects(tmp_path):
    c = client(tmp_path)
    assert c.get("/", follow_redirects=False).headers["location"] == "/ui/"
    page = c.get("/ui/")
    assert page.status_code == 200 and "Media Intelligence" in page.text and "cytoscape" in page.text
    assert c.get("/ui/app.js").status_code == 200 and c.get("/ui/style.css").status_code == 200
    # The graph API is untouched and the helpers stay out of the OpenAPI schema.
    paths = c.get("/openapi.json").json()["paths"]
    assert {"/entity/{name}/network", "/connections/new", "/entities/central"} <= set(paths)
    assert not any(p.startswith(("/ui", "/ui-api")) for p in paths)


def test_overview_reports_config_and_empty_graph(tmp_path):
    data = client(tmp_path).get("/ui-api/overview").json()
    assert data["graph"]["available"] and data["graph"]["counts"]["nodes"] == 0 and data["graph"]["runs"] == []
    assert {s["name"] for s in data["sources"]["items"]} == {"aljazeera", "hackernews", "mastodon"}


def test_overview_without_database_explains_instead_of_failing(tmp_path):
    data = client(tmp_path, migrate=False).get("/ui-api/overview").json()
    assert data["graph"]["available"] is False and data["graph"]["message"]


def test_pipeline_runner_runs_the_unchanged_cli_once_at_a_time(tmp_path, monkeypatch):
    started = []

    class FakeProcess:
        def __init__(self, args, stdout, stderr, env):
            started.append((args, env))
            stdout.write("fetched source=hackernews depth=0\nnavigation_blocked url=https://ads.example\n")
            self.code = None

        def poll(self):
            return self.code

    monkeypatch.setattr("media_intelligence.ui.subprocess.Popen", FakeProcess)
    c = client(tmp_path)
    assert c.get("/ui-api/pipeline").json() == {"state": "idle"}
    first = c.post("/ui-api/pipeline").json()
    assert first["state"] == "running"
    assert first["log_tail"] == ["fetched source=hackernews depth=0"]  # browser noise filtered out
    assert c.post("/ui-api/pipeline").status_code == 409
    args, env = started[0]
    assert args == [sys.executable, "-m", "media_intelligence.cli", "ingest"]
    assert env["MI_DATABASE_PATH"].endswith("graph.sqlite3")
    c.app.state.pipeline.process.code = 3
    done = c.get("/ui-api/pipeline").json()
    assert done["state"] == "finished" and done["exit_code"] == 3 and done["exit_meaning"].startswith("partial")
    assert c.post("/ui-api/pipeline").status_code == 200 and len(started) == 2
