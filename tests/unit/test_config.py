"""Network-free tests for configuration boundaries and the crawl frontier."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from media_intelligence.config import CrawlLimits, PipelineConfig, SourceProfile, canonical_url
from media_intelligence.crawler import Frontier


def source(name="news", host="example.com"):
    return SourceProfile(name=name, source_type="news", adapter="article",
                         allowed_domains=[host], seeds=[f"https://{host}/article"])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost/", "http://127.0.0.1/",
                                  "http://169.254.169.254/", "https://user:secret@example.com/",
                                  "https://example.com:8080/", "http://[::1]/"])
def test_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        canonical_url(url)


def test_url_identity_preserves_meaningful_queries():
    assert canonical_url("https://EXAMPLE.com/story?id=7&utm_source=test#part") == "https://example.com/story?id=7"
    assert canonical_url("https://example.com/story?id=8") != canonical_url("https://example.com/story?id=7")


def test_exact_host_match_and_out_of_policy_seed():
    profile = source()
    assert profile.accepts("https://example.com/new-seed")
    assert not profile.accepts("https://example.com.evil.test/article")
    assert not profile.accepts("https://sub.example.com/article")
    with pytest.raises(ValidationError):
        SourceProfile(name="news", source_type="news", adapter="article",
                      allowed_domains=["example.com"], seeds=["https://other.test/page"])


def test_frontier_depth_and_cycles():
    frontier = Frontier(PipelineConfig(crawl=CrawlLimits(max_depth=1), sources=[source()]))
    root = frontier.pop()
    assert root[2] == 0
    frontier.add("https://example.com/child", 1)
    frontier.add("https://example.com/child#again", 1)
    frontier.add("https://example.com/grandchild", 2)
    frontier.add("https://other.test/outside", 1)
    assert frontier.pop()[0] == "https://example.com/child"
    assert frontier.pop() is None
    assert frontier.skips["depth_limit"] == 1


def test_frontier_depth_zero_and_source_fairness():
    first = source()
    first.seeds.append("https://example.com/second")
    config = PipelineConfig(crawl=CrawlLimits(max_depth=0, max_pages_total=3),
                            sources=[first, source("other", "other.test")])
    frontier = Frontier(config)
    assert frontier.pop()[1].name == "news"
    frontier.add("https://example.com/child", 1)
    assert frontier.pop()[1].name == "other"  # round-robin: the second source's seed before the first's second
    assert frontier.pop()[0] == "https://example.com/second"
    assert frontier.pop() is None and frontier.skips["depth_limit"] == 1


def test_more_seeds_than_a_page_budget_is_a_configuration_error():
    """A seed past max_pages_per_source / max_pages_total would otherwise be skipped silently."""
    many = source().model_copy(update={"seeds": [f"https://example.com/a{i}" for i in range(3)]})
    with pytest.raises(ValidationError, match="news has 3 seeds but crawl.max_pages_per_source is 2"):
        PipelineConfig(crawl=CrawlLimits(max_pages_per_source=2), sources=[many])
    with pytest.raises(ValidationError, match="4 seeds in total but crawl.max_pages_total is 3"):
        PipelineConfig(crawl=CrawlLimits(max_pages_total=3), sources=[many, source("other", "other.test")])


def test_out_of_policy_seed_error_names_the_seed():
    with pytest.raises(ValidationError, match=r"listing_paths: https://example.com/news/\. Add"):
        SourceProfile(name="news", source_type="news", adapter="article", allowed_domains=["example.com"],
                      include_paths=["^/news/[0-9]+$"], seeds=["https://example.com/news/1", "https://example.com/news/"])


def test_seed_stays_with_the_source_that_declared_it():
    """Two profiles on one host: a seed is crawled under its own profile, not the first one that matches."""
    news = SourceProfile(name="news", source_type="news", adapter="article", allowed_domains=["example.com"],
                         seeds=["https://example.com/a"])
    blog = SourceProfile(name="blog", source_type="analysis", adapter="article", allowed_domains=["example.com"],
                         seeds=["https://example.com/b"])
    frontier = Frontier(PipelineConfig(sources=[news, blog]))
    assert [(url, src.name) for url, src, _ in (frontier.pop(), frontier.pop())] == [
        ("https://example.com/a", "news"), ("https://example.com/b", "blog")]


def test_shipped_config_validates_with_three_source_types():
    from media_intelligence.config import load_config
    config = load_config(Path(__file__).resolve().parents[2] / "config" / "sources.toml")
    assert {s.source_type for s in config.sources} == {"news", "discussion", "microblog"}
    assert {s.adapter for s in config.sources} == {"article", "hackernews", "mastodon"}



def test_per_source_depth_and_delay_only_tighten_global_limits():
    shallow = source("shallow", "shallow.test").model_copy(update={"max_depth": 0, "min_delay_seconds": 30})
    deep = source("deep", "deep.test").model_copy(update={"max_depth": 5, "min_delay_seconds": 0})
    config = PipelineConfig(crawl=CrawlLimits(max_depth=1, delay_per_host_seconds=2), sources=[shallow, deep])
    assert (config.depth_limit(shallow), config.delay_for(shallow)) == (0, 30)
    assert (config.depth_limit(deep), config.delay_for(deep)) == (1, 2)
    frontier = Frontier(config)
    frontier.add("https://shallow.test/permalink", 1)
    frontier.add("https://deep.test/child", 1)
    assert [frontier.pop()[0] for _ in range(3)] == [
        "https://shallow.test/article", "https://deep.test/article", "https://deep.test/child"]
    assert frontier.skips["depth_limit"] == 1


def test_listing_seed_followed_but_thread_links_and_discovered_listings_are_not():
    """HN-style policy: listings are entry points; thread pages only link to their own permalinks."""
    hn = SourceProfile(name="hn", source_type="discussion", adapter="hackernews", allowed_domains=["news.ycombinator.com"],
                       include_paths=["^/item$"], listing_paths=["^/$", "^/(news|best)$"], follow_links="listings",
                       seeds=["https://news.ycombinator.com/", "https://news.ycombinator.com/item?id=1"])
    frontier = Frontier(PipelineConfig(crawl=CrawlLimits(max_depth=1), sources=[hn]))
    assert [frontier.pop()[0] for _ in range(2)] == ["https://news.ycombinator.com/", "https://news.ycombinator.com/item?id=1"]
    frontier.follow(hn, "https://news.ycombinator.com/item?id=1", ["https://news.ycombinator.com/item?id=5"], 0)
    frontier.follow(hn, "https://news.ycombinator.com/", ["https://news.ycombinator.com/item?id=2",
                                                          "https://news.ycombinator.com/news?p=2"], 0)
    assert frontier.pop()[0] == "https://news.ycombinator.com/item?id=2"
    assert frontier.pop() is None
    assert frontier.skips["links_not_followed"] == 1 and frontier.skips["listing_not_seed"] == 1


def test_every_seed_swapped_needs_no_code_change(tmp_path):
    """Evaluator scenario: replace every source's seeds with other valid pages of the whitelisted sites."""
    import re
    import shutil

    from media_intelligence.config import load_config
    swaps = {"aljazeera": ["https://www.aljazeera.com/news/2026/10/7/kyiv-attack", "https://www.aljazeera.com/middle-east/"],
             "hackernews": ["https://news.ycombinator.com/", "https://news.ycombinator.com/item?id=49872723"],
             "mastodon": ["https://mastodon.social/@404mediaco/117378117850915865"]}
    blocks = Path("config/sources.toml").read_text().split("[[sources]]")
    for i, block in enumerate(blocks[1:], 1):
        name = re.search(r'^name = "([^"]+)"', block, re.MULTILINE).group(1)
        seeds = ", ".join(f'"{url}"' for url in swaps[name])
        blocks[i] = re.sub(r"^seeds = \[.*?\]", f"seeds = [{seeds}]", block, count=1, flags=re.DOTALL | re.MULTILINE)
    for name in ("aliases.toml", "topics.toml"):
        shutil.copy(Path("config") / name, tmp_path / name)
    (tmp_path / "sources.toml").write_text("[[sources]]".join(blocks))
    config = load_config(tmp_path / "sources.toml")
    assert {s.name: s.seeds for s in config.sources} == swaps
