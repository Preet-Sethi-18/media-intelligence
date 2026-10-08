"""Review regressions for normalization: block-aware text, logical thread URLs, and article headline segments.

All HTML below is labelled synthetic: written for these tests in the shape of the real pages, never crawled.
"""

from datetime import UTC, datetime

import pytest
from bs4 import BeautifulSoup

from media_intelligence.config import CrawlLimits, SourceProfile
from media_intelligence.models import FetchResult
from media_intelligence.normalization import NormalizationError, normalize, text_of, thread_url

SOURCE_TYPES = {"article": "news", "reddit": "discussion", "hackernews": "discussion"}
FILLER = " The statement was read out at a press conference and published in full afterwards." * 2


def normalized(html, adapter, url, allowed):
    source = SourceProfile(name="test", source_type=SOURCE_TYPES[adapter], adapter=adapter, seeds=[url],
                           allowed_domains=[allowed])
    page = FetchResult(requested_url=url, final_url=url, source_name="test", depth=0,
                       fetched_at=datetime(2026, 10, 7, tzinfo=UTC), success=True, status_code=200, html=html)
    return normalize(page, source, CrawlLimits())


def test_inline_links_stay_in_their_sentence_and_blocks_become_lines():
    html = """<article><p>Israeli Prime Minister
      <a href="/n">Benjamin Netanyahu</a> has met US President <a href="/b">Joe Biden</a> at the
      <a href="/w">White House</a>, where they discussed <a href="/s">sanctions</a> on Iran.</p>
      <!-- synthetic hidden note --><ul><li>First point</li><li>Second<br>line</li></ul>
      <pre>keep   this
spacing</pre></article>"""
    sentence = ("Israeli Prime Minister Benjamin Netanyahu has met US President Joe Biden at the White House, "
                "where they discussed sanctions on Iran.")
    assert text_of(BeautifulSoup(html, "html.parser").article).split("\n") == [
        sentence, "First point", "Second", "line", "keep this", "spacing"]


@pytest.mark.parametrize("url", [
    "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/synthetic_slug/",
    "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/synthetic_slug/?sort=new",
    "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/comment/nm1abcd/",
    "https://old.reddit.com/r/GeoPolitics/comments/1ojf0xi/",
])
def test_reddit_permalink_variants_share_the_thread_url(url):
    assert thread_url("reddit", url) == "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/"


def test_hackernews_item_url_keeps_only_the_story_id():
    assert thread_url("hackernews", "https://news.ycombinator.com/item?id=100&p=2") == \
        "https://news.ycombinator.com/item?id=100"
    assert thread_url("article", "https://example.com/a?page=2") == "https://example.com/a?page=2"


def test_reddit_item_stores_the_thread_url_and_keeps_the_fetched_url():
    html = """<shreddit-post id="t3_1ojf0xi" post-title="Synthetic thread" author="poster">
      <div slot="text-body">Synthetic post body.</div></shreddit-post>"""
    item = normalized(html, "reddit", "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/comment/nm1abcd/",
                      "www.reddit.com")
    assert item.source_url == "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/"
    assert item.metadata["fetched_url"] == "https://www.reddit.com/r/geopolitics/comments/1ojf0xi/comment/nm1abcd/"
    assert item.requested_url.endswith("/comment/nm1abcd/")


def test_hackernews_comment_permalink_page_is_not_a_thread():
    # Synthetic comment permalink: its top row is a plain tr.athing with the requested id but no story title line.
    html = """<table><tr class="athing" id="101"><td><span class="comhead"><a class="hnuser">alice</a>
      <span class="onstory"> | on: <a href="item?id=100">Synthetic story</a></span></span>
      <div class="commtext">A synthetic comment.</div></td></tr></table>"""
    with pytest.raises(NormalizationError, match="unsupported_structure"):
        normalized(html, "hackernews", "https://news.ycombinator.com/item?id=101", "news.ycombinator.com")


def test_hackernews_story_row_without_submission_class_still_works():
    html = """<table><tr class="athing" id="100"><td><span class="titleline"><a href="https://example.com/x">
      Synthetic story title</a></span></td></tr></table>"""
    item = normalized(html, "hackernews", "https://news.ycombinator.com/item?id=100&p=2", "news.ycombinator.com")
    assert item.title == "Synthetic story title" and item.source_url == "https://news.ycombinator.com/item?id=100"


def test_article_headline_is_its_own_segment_and_not_part_of_the_body():
    html = ("""<script type="application/ld+json">{"@type": "NewsArticle",
        "headline": "Putin and Zelensky hold talks in Geneva | Synthetic Daily"}</script><article><p>"""
            + "Officials described the session as constructive." + FILLER + "</p></article>")
    item = normalized(html, "article", "https://example.com/story", "example.com")
    assert [(s.id, s.kind) for s in item.segments] == [("title", "title"), ("body", "body")]
    assert item.segments[0].text == "Putin and Zelensky hold talks in Geneva"  # site suffix dropped
    assert item.title == "Putin and Zelensky hold talks in Geneva | Synthetic Daily"  # metadata kept verbatim
    assert "Putin" not in item.body


def test_article_headline_repeated_in_the_body_is_not_duplicated():
    html = ("<h1>Ceasefire talks resume</h1><article><p>Ceasefire talks resume in Doha." + FILLER + "</p></article>")
    item = normalized(html, "article", "https://example.com/story", "example.com")
    assert [s.id for s in item.segments] == ["body"]
