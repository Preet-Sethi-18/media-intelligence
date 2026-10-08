"""Command-line checks and the no-hardcoded-URL rule (evaluation PDF 1a: seeds live in configuration only)."""

import json
import re
import shutil
from pathlib import Path

from media_intelligence import cli

ROOT = Path(__file__).resolve().parents[2]


def test_no_url_literal_in_source_code():
    """Seeds and crawl targets come from config/sources.toml; the package contains no literal http(s) URL."""
    for path in (ROOT / "src" / "media_intelligence").glob("*.py"):
        assert not re.search(r"https?://[A-Za-z0-9]", path.read_text(encoding="utf-8")), path.name


def test_validate_config_checks_the_vocabulary_files_ingest_needs(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MI_REPORT_DIR", str(tmp_path / "reports"))
    shutil.copy(ROOT / "config" / "sources.toml", tmp_path)
    # Without aliases.toml/topics.toml next to it, ingest would refuse the config, so validation must too.
    assert cli.main(["validate-config", "--config", str(tmp_path / "sources.toml")]) == 2
    assert "aliases.toml" in capsys.readouterr().err
    for name in ("aliases.toml", "topics.toml"):
        shutil.copy(ROOT / "config" / name, tmp_path)
    assert cli.main(["validate-config", "--config", str(tmp_path / "sources.toml")]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["valid"] and summary["source_types"] == ["discussion", "microblog", "news"]
    assert summary["alias_entries"] > 0 and summary["topics"] > 0
    (tmp_path / "topics.toml").write_text('[[topics]]\nname = "a"\npatterns = ["x"]\n[[topics]]\nname = "b"\n'
                                          'patterns = ["X"]\n')
    assert cli.main(["validate-config", "--config", str(tmp_path / "sources.toml")]) == 2
    assert "more than one topic" in capsys.readouterr().err
