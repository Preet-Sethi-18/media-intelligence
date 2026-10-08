"""Validated TOML configuration and environment settings."""

import ipaddress
import re
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import SourceType


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    if parts.username or parts.password:
        raise ValueError("Credentials are not allowed inside URLs")
    host = parts.hostname.rstrip(".").encode("idna").decode().lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Local addresses are not crawl targets")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError("Private or reserved addresses are not crawl targets")
    port = parts.port
    if port not in {None, 80, 443}:
        raise ValueError("Only standard web ports are supported")
    netloc = f"[{host}]" if ":" in host else host
    if port and not (parts.scheme == "http" and port == 80 or parts.scheme == "https" and port == 443):
        netloc += f":{port}"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
    return urlunsplit((parts.scheme, netloc, parts.path or "/", urlencode(sorted(query)), ""))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CrawlLimits(StrictModel):
    max_depth: int = Field(default=1, ge=0, le=5)
    max_pages_total: int = Field(default=30, ge=1, le=1000)
    max_pages_per_source: int = Field(default=10, ge=1, le=500)
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=1, ge=0, le=3)
    delay_per_host_seconds: float = Field(default=2, ge=0.2, le=60)
    max_body_characters: int = Field(default=100_000, ge=100, le=500_000)
    max_comments_per_thread: int = Field(default=100, ge=1, le=500)
    respect_robots: bool = True
    save_raw: bool = False


class SourceProfile(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    source_type: SourceType
    adapter: Literal["article", "reddit", "x", "forum", "hackernews", "mastodon"]
    seeds: list[str] = Field(min_length=1)
    allowed_domains: list[str] = Field(min_length=1)
    include_paths: list[str] = Field(default_factory=list)
    # Section/index pages: crawled so their links are followed, never stored as content items.
    listing_paths: list[str] = Field(default_factory=list)
    body_selector: str | None = None
    # Blocks inside the body that belong to the page, not the item: related-story lists, newsletter forms, ads.
    exclude_selectors: list[str] = Field(default_factory=list)
    wait_for: str | None = None
    render_wait_seconds: float = Field(default=2, ge=0, le=15)
    # Per-source overrides: a shallower depth (e.g. comment permalinks that only re-show one thread)
    # and a longer politeness delay (e.g. a robots.txt Crawl-delay). Neither can loosen global limits.
    max_depth: int | None = Field(default=None, ge=0, le=5)
    min_delay_seconds: float | None = Field(default=None, ge=0, le=120)
    # "listings": follow links only from listing pages (thread pages link mostly to their own comment permalinks).
    follow_links: Literal["all", "listings"] = "all"

    @field_validator("allowed_domains")
    @classmethod
    def exact_hosts(cls, values: list[str]) -> list[str]:
        result = []
        for value in values:
            value = value.strip().lower().rstrip(".")
            if not value or any(c in value for c in "/:*@?#"):
                raise ValueError("Allowed domains must be exact hostnames, without schemes or wildcards")
            canonical_url(f"https://{value}/")
            result.append(value.encode("idna").decode())
        return result

    @field_validator("seeds")
    @classmethod
    def valid_seeds(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(canonical_url(v) for v in values))

    @field_validator("include_paths", "listing_paths")
    @classmethod
    def valid_patterns(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 200:
                raise ValueError("Path patterns must be at most 200 characters")
            re.compile(value)
        return values

    def accepts(self, url: str, *, check_path: bool = True) -> bool:
        try:
            parts = urlsplit(canonical_url(url))
        except ValueError:
            return False
        return parts.hostname in self.allowed_domains and (
            not check_path or not self.include_paths
            or any(re.search(pattern, parts.path) for pattern in self.include_paths + self.listing_paths)
        )

    def is_listing(self, url: str) -> bool:
        path = urlsplit(canonical_url(url)).path
        return any(re.search(pattern, path) for pattern in self.listing_paths)

    @model_validator(mode="after")
    def seeds_in_policy(self):
        if outside := [url for url in self.seeds if not self.accepts(url)]:
            raise ValueError(f"Seeds for {self.name} outside its allowed_domains/include_paths/listing_paths: "
                             f"{', '.join(outside)}. Add the host to allowed_domains and a matching path pattern to "
                             f"include_paths (content pages) or listing_paths (section/front pages).")
        return self


class PipelineConfig(StrictModel):
    crawl: CrawlLimits = Field(default_factory=CrawlLimits)
    sources: list[SourceProfile] = Field(min_length=1)
    aliases_file: str = "aliases.toml"
    topics_file: str = "topics.toml"

    @model_validator(mode="after")
    def distinct_names(self):
        names = [source.name for source in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("Source names must be unique")
        if self.crawl.max_pages_total < len(self.sources):
            raise ValueError("Total page budget must allow at least one page per source")
        # A seed beyond a page budget would be skipped silently by the frontier; refuse the configuration instead.
        for source in self.sources:
            if len(source.seeds) > self.crawl.max_pages_per_source:
                raise ValueError(f"Source {source.name} has {len(source.seeds)} seeds but crawl.max_pages_per_source "
                                 f"is {self.crawl.max_pages_per_source}; raise it or remove seeds")
        seeds = sum(len(source.seeds) for source in self.sources)
        if seeds > self.crawl.max_pages_total:
            raise ValueError(f"The sources have {seeds} seeds in total but crawl.max_pages_total is "
                             f"{self.crawl.max_pages_total}; raise it or remove seeds")
        return self

    def source_for(self, url: str) -> SourceProfile | None:
        return next((s for s in self.sources if s.accepts(url)), None)

    def depth_limit(self, source: SourceProfile) -> int:
        return min(self.crawl.max_depth, source.max_depth if source.max_depth is not None else self.crawl.max_depth)

    def delay_for(self, source: SourceProfile) -> float:
        return max(self.crawl.delay_per_host_seconds, source.min_delay_seconds or 0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MI_", env_file=".env", extra="ignore")
    database_path: Path = Path("data/media_intelligence.sqlite3")
    config_path: Path = Path("config/sources.toml")
    report_dir: Path = Path("reports")
    spacy_model: str = "en_core_web_sm"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    browser_state_path: Path | None = None
    browser_headless: bool = True
    growth_min_delta: int = Field(default=3, ge=1)
    growth_min_ratio: float = Field(default=0.5, gt=0)


def load_config(path: Path) -> PipelineConfig:
    with path.open("rb") as stream:
        return PipelineConfig.model_validate(tomllib.load(stream))

