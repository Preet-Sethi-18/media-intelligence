"""Crawl4AI fetches with a bounded, testable breadth-first URL frontier."""

import asyncio
import ipaddress
import logging
import os
import socket
import time
from collections import Counter, deque
from urllib.parse import urljoin, urlsplit

from .config import PipelineConfig, Settings, canonical_url
from .models import FetchResult

log = logging.getLogger(__name__)


class Frontier:
    """Seeds are depth zero. Every queued link is classified and validated."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.queue = deque()
        self.seen: set[str] = set()
        self.counts: Counter = Counter()
        self.skips: Counter = Counter()
        # Round-robin seed order avoids starving the later source profiles.
        for i in range(max(len(s.seeds) for s in config.sources)):
            for source in config.sources:
                if i < len(source.seeds):
                    self.add(source.seeds[i], 0, source)

    def add(self, url: str, depth: int, declared=None):
        """Queue a URL under the source that declared it (a seed) or the first source whose policy accepts it."""
        if depth > self.config.crawl.max_depth:
            self.skips["depth_limit"] += 1
            return
        try:
            url = canonical_url(url)
        except ValueError:
            self.skips["invalid_url"] += 1
            return
        source = declared if declared is not None and declared.accepts(url) else self.config.source_for(url)
        if source is None:
            self.skips["outside_source_policy"] += 1
            return
        if depth > self.config.depth_limit(source):
            self.skips["depth_limit"] += 1
            return
        if depth > 0 and source.is_listing(url):
            # Listing pages are entry points only; discovered "next page"/section links would spend the budget.
            self.skips["listing_not_seed"] += 1
            return
        if url in self.seen:
            self.skips["duplicate_url"] += 1
            return
        self.seen.add(url)
        self.queue.append((url, source, depth))

    def follow(self, source, page_url: str, links: list[str], depth: int):
        if source.follow_links == "listings" and not source.is_listing(page_url):
            self.skips["links_not_followed"] += len(links)
            return
        for link in links:
            self.add(link, depth + 1)

    def pop(self):
        if sum(self.counts.values()) >= self.config.crawl.max_pages_total:
            self.skips["total_page_limit"] += len(self.queue)
            self.queue.clear()
            return None
        while self.queue:
            url, source, depth = self.queue.popleft()
            if self.counts[source.name] >= self.config.crawl.max_pages_per_source:
                self.skips["source_page_limit"] += 1
                continue
            self.counts[source.name] += 1
            return url, source, depth
        return None


class BrowserCrawler:
    def __init__(self, config: PipelineConfig, settings: Settings):
        # Checked here, not when crawling starts, so a bad path is a setup error before any run is recorded.
        state = settings.browser_state_path
        if state and not state.is_file():
            raise ValueError("MI_BROWSER_STATE_PATH must point to an existing local storage-state JSON file")
        self.config = config
        self.settings = settings
        self.frontier = Frontier(config)
        self._dns: dict[str, bool] = {}
        self._last_fetch: dict[str, float] = {}

    async def public_destination(self, url: str) -> bool:
        try:
            host = urlsplit(canonical_url(url)).hostname
        except ValueError:
            return False
        if host in self._dns:
            return self._dns[host]
        try:
            entries = await asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
            allowed = bool(entries) and all(ipaddress.ip_address(e[4][0]).is_global for e in entries)
        except (OSError, ValueError):
            allowed = False
        self._dns[host] = allowed
        return allowed

    async def _guard(self, page, context, **kwargs):
        async def route_request(route):
            request = route.request
            url = request.url
            if request.resource_type in {"image", "media", "font"}:
                await route.abort()
                return
            if not await self.public_destination(url):
                await route.abort()
                return
            if request.is_navigation_request() and self.config.source_for(url) is None:
                log.warning("navigation_blocked url=%s", urlsplit(url)._replace(query="").geturl())
                await route.abort()
                return
            await route.continue_()

        await page.route("**/*", route_request)
        return page

    async def pages(self):
        cache_dir = self.settings.database_path.parent / "crawler-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(cache_dir.resolve()))
        # Lazy import keeps configuration checks and API startup free of browser work.
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

        state = self.settings.browser_state_path
        browser = BrowserConfig(headless=self.settings.browser_headless, verbose=False,
                                storage_state=str(state) if state else None,
                                ignore_https_errors=False, accept_downloads=False)
        async with AsyncWebCrawler(config=browser, base_directory=str(cache_dir)) as crawler:
            crawler.crawler_strategy.set_hook("on_page_context_created", self._guard)
            while item := self.frontier.pop():
                url, source, depth = item
                if not await self.public_destination(url):
                    yield FetchResult(requested_url=url, final_url=url, source_name=source.name,
                                      depth=depth, error="URL did not resolve to a public destination")
                    continue
                host = urlsplit(url).hostname
                delay = self.config.delay_for(source) - (time.monotonic() - self._last_fetch.get(host, 0))
                if delay > 0:
                    await asyncio.sleep(delay)
                run = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, verbose=False,
                                       page_timeout=self.config.crawl.request_timeout_seconds * 1000,
                                       wait_until="domcontentloaded", wait_for=source.wait_for,
                                       delay_before_return_html=source.render_wait_seconds,
                                       word_count_threshold=1, check_robots_txt=self.config.crawl.respect_robots)
                fetched = None
                for attempt in range(self.config.crawl.max_retries + 1):
                    self._last_fetch[host] = time.monotonic()
                    try:
                        result = await asyncio.wait_for(crawler.arun(url=url, config=run),
                                                        self.config.crawl.request_timeout_seconds + 20)
                        final_url = getattr(result, "redirected_url", None) or result.url or url
                        in_policy = self.config.source_for(final_url) is not None
                        links = []
                        for group in (result.links or {}).values():
                            if isinstance(group, list):
                                links.extend(urljoin(final_url, link["href"]) for link in group
                                             if isinstance(link, dict) and link.get("href"))
                        fetched = FetchResult(requested_url=url, final_url=final_url, source_name=source.name,
                                              depth=depth, success=bool(result.success and in_policy),
                                              status_code=result.status_code, html=result.html or "", links=links,
                                              error=(result.error_message or None) if in_policy else "Redirect left source policy")
                    except Exception as exc:  # noqa: BLE001 - any crawl error is a recorded per-page failure
                        fetched = FetchResult(requested_url=url, final_url=url, source_name=source.name,
                                              depth=depth, error=f"{type(exc).__name__}: {str(exc)[:250]}")
                    if fetched.success or fetched.status_code in {401, 403, 404, 410}:
                        break
                    if attempt < self.config.crawl.max_retries:
                        await asyncio.sleep(max(self.config.delay_for(source), 2 ** attempt))
                log.info("fetched source=%s depth=%d status=%s success=%s url=%s",
                         source.name, depth, fetched.status_code, fetched.success, url)
                yield fetched
                if fetched.success:
                    self.frontier.follow(source, fetched.final_url, fetched.links, depth)

