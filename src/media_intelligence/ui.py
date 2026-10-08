"""Browser interface: a static page plus helper routes. The graph API and the pipeline are unchanged.

The page calls the existing endpoints for graph data. The helpers here only read the database and config
(overview) or start the unchanged `ingest` command in a separate process (pipeline runner).
"""

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import analysis, db
from .config import Settings, load_config
from .models import timestamp, utcnow

STATIC = Path(__file__).parent / "static"
EXIT_MEANING = {0: "complete", 1: "fatal error", 2: "setup or configuration error", 3: "partial: a source or seed failed",
                130: "interrupted"}
# Browser navigation noise from ad/tracker frames; useful in a debug log, not in a progress view.
_NOISE = ("navigation_blocked",)


class PipelineRunner:
    """One `media-intelligence ingest` subprocess at a time, with its output kept for the progress view."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.started_at = self.finished_at = None
        self.exit_code: int | None = None
        self.log_path = Path(settings.report_dir) / "ui-ingest.log"

    def start(self) -> bool:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return False
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ | {"MI_DATABASE_PATH": str(self.settings.database_path),
                                "MI_CONFIG_PATH": str(self.settings.config_path),
                                "MI_REPORT_DIR": str(self.settings.report_dir)}
            with self.log_path.open("w") as log:
                # The same CLI an operator runs by hand; the config file decides what is crawled.
                self.process = subprocess.Popen([sys.executable, "-m", "media_intelligence.cli", "ingest"],
                                                stdout=log, stderr=subprocess.STDOUT, env=env)
            self.started_at, self.finished_at, self.exit_code = timestamp(utcnow()), None, None
            return True

    def status(self) -> dict:
        with self.lock:
            if self.process is None:
                return {"state": "idle"}
            code = self.process.poll()
            if code is not None and self.exit_code is None:
                self.exit_code, self.finished_at = code, timestamp(utcnow())
            lines = deque(maxlen=40)
            if self.log_path.exists():
                for line in self.log_path.read_text(errors="replace").splitlines():
                    if line.strip() and not any(noise in line for noise in _NOISE):
                        lines.append(line[:300])
            return {"state": "running" if code is None else "finished", "started_at": self.started_at,
                    "finished_at": self.finished_at, "exit_code": self.exit_code,
                    "exit_meaning": EXIT_MEANING.get(self.exit_code) if self.exit_code is not None else None,
                    "log_tail": list(lines)}


def _sources(settings: Settings) -> dict:
    try:
        config = load_config(settings.config_path)
    except (OSError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}", "items": []}
    return {"path": str(settings.config_path), "max_depth": config.crawl.max_depth,
            "max_pages_per_source": config.crawl.max_pages_per_source,
            "items": [{"name": s.name, "source_type": s.source_type, "adapter": s.adapter,
                       "allowed_domains": s.allowed_domains, "seeds": s.seeds} for s in config.sources]}


def _graph(settings: Settings) -> dict:
    try:
        conn = db.connect(settings.database_path, readonly=True)
    except db.SchemaError as exc:
        return {"available": False, "message": str(exc)}
    try:
        db.check_schema(conn)
        relations = dict(conn.execute("SELECT relation_type, COUNT(*) FROM edges GROUP BY 1").fetchall())
        runs = []
        for row in conn.execute("SELECT id, status, started_at, finished_at, summary_json FROM crawl_runs "
                                "ORDER BY id DESC"):
            summary = json.loads(row["summary_json"] or "{}")
            runs.append({"id": row["id"], "status": row["status"], "started_at": row["started_at"],
                         "finished_at": row["finished_at"], "stored_new": summary.get("stored_new"),
                         "stored_unchanged": summary.get("stored_unchanged"),
                         "failed_seeds": summary.get("failed_seeds") or {},
                         "missing_sources": summary.get("missing_sources") or []})
        return {"available": True, "counts": analysis.graph_counts(conn),
                "typed_edges": sum(n for rel, n in relations.items() if rel != "mentioned_with"),
                "weak_edges": relations.get("mentioned_with", 0), "relations": relations, "runs": runs}
    except db.SchemaError as exc:
        return {"available": False, "message": str(exc)}
    finally:
        conn.close()


def install(app: FastAPI, settings: Settings) -> None:
    runner = PipelineRunner(settings)
    app.state.pipeline = runner

    @app.get("/", include_in_schema=False)
    def home():
        return RedirectResponse("/ui/")

    @app.get("/ui-api/overview", include_in_schema=False)
    def overview():
        return {"database": str(settings.database_path), "graph": _graph(settings), "sources": _sources(settings)}

    @app.get("/ui-api/pipeline", include_in_schema=False)
    def pipeline_status():
        return runner.status()

    @app.post("/ui-api/pipeline", include_in_schema=False)
    def pipeline_start():
        if not runner.start():
            return JSONResponse({"error": {"code": "pipeline_running", "message": "A pipeline run is already in progress"}},
                                status_code=409)
        return runner.status()

    app.mount("/ui", StaticFiles(directory=STATIC, html=True), name="ui")
