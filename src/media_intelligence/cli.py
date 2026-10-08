"""Command-line entry points. Crawl failures have explicit, nonzero exit status."""

import argparse
import asyncio
import json
import logging
import sys
from hashlib import sha256
from pathlib import Path

from .config import Settings, load_config
from .models import timestamp, utcnow

logger = logging.getLogger(__name__)


async def probe(config, settings, *, save_raw=False):
    from .crawler import BrowserCrawler
    from .normalization import NormalizationError, normalize

    config = config.model_copy(deep=True)
    config.crawl.max_depth = 0
    crawler = BrowserCrawler(config, settings)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    folder = settings.report_dir / ("probe-" + utcnow().strftime("%Y%m%dT%H%M%SZ"))
    folder.mkdir()
    report = {"started_at": timestamp(utcnow()), "results": [], "kind": "live_crawl4ai_probe"}
    for_source = {s.name: s for s in config.sources}
    async for page in crawler.pages():
        row = page.model_dump(mode="json", exclude={"html", "links"})
        row["html_bytes"] = len(page.html.encode())
        key = page.source_name + "-" + sha256(page.requested_url.encode()).hexdigest()[:12]
        if save_raw and page.html:
            (folder / (key + ".html")).write_text(page.html)
        try:
            declared = for_source[page.source_name]
            profile = declared if declared.accepts(page.final_url) else config.source_for(page.final_url) or declared
            content = normalize(page, profile, config.crawl)
            row.update(normalized=True, title=content.title, body_characters=len(content.body),
                       segments=len(content.segments), warnings=content.metadata["warnings"])
            (folder / (key + ".json")).write_text(content.model_dump_json(indent=2))
        except NormalizationError as exc:
            row.update(normalized=False, normalization_error=str(exc))
        report["results"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    passed = {row["source_name"] for row in report["results"] if row["normalized"]}
    report.update(finished_at=timestamp(utcnow()), passed_sources=sorted(passed),
                  missing_sources=sorted(set(for_source) - passed))
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Report: {folder / 'report.json'}")
    return 0 if not report["missing_sources"] else 3


def vocabulary(path: Path, config) -> tuple[int, int]:
    """Check the alias and topic files exactly as ingest will: relative names resolve next to sources.toml."""
    import tomllib

    from .extraction import load_aliases, load_topics

    files = (path.parent / config.aliases_file, path.parent / config.topics_file)
    if missing := [str(file) for file in files if not file.is_file()]:
        raise ValueError(f"Required configuration file not found: {', '.join(missing)}")
    entries, _, _ = load_aliases(tomllib.loads(files[0].read_text(encoding="utf-8")))
    return len(entries), len(load_topics(tomllib.loads(files[1].read_text(encoding="utf-8"))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evidence-backed media intelligence")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-config", "probe", "ingest"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path)
        if name == "probe":
            command.add_argument("--save-raw", action="store_true", help="Save local HTML for diagnosis; never commit sessions")
    args = parser.parse_args(argv)
    try:
        settings = Settings()
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        path = args.config or settings.config_path
        config = load_config(path)
        if args.command == "validate-config":
            aliases, topics = vocabulary(path, config)
            print(json.dumps({"valid": True, "sources": [s.name for s in config.sources],
                              "source_types": sorted({s.source_type for s in config.sources}),
                              "seed_count": sum(len(s.seeds) for s in config.sources),
                              "max_depth": config.crawl.max_depth, "alias_entries": aliases,
                              "topics": topics}, indent=2))
            return 0
        if args.command == "probe":
            return asyncio.run(probe(config, settings, save_raw=args.save_raw))
        from .pipeline import ingest
        return asyncio.run(ingest(config, settings, path))
    except (ValueError, OSError) as exc:
        print(f"Configuration/setup error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; committed items remain safe.", file=sys.stderr)
        return 130
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
