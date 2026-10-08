"""Labelled synthetic HTML isolates adapter behavior; it is not live crawl evidence."""

from datetime import UTC, datetime

import pytest

from media_intelligence.config import CrawlLimits, SourceProfile
from media_intelligence.models import FetchResult
from media_intelligence.normalization import NormalizationError, normalize


def normalized(html, adapter="article", url="https://example.com/story", **limits):
    source_type = {"article": "news", "reddit": "discussion", "x": "microblog", "hackernews": "discussion",
                   "mastodon": "microblog"}[adapter]
    source = SourceProfile(name="test", source_type=source_type, adapter=adapter,
                           seeds=[url], allowed_domains=[url.split("/")[2]])
    page = FetchResult(requested_url=url, final_url=url, source_name="test", depth=0,
                       fetched_at=datetime(2026, 10, 7, tzinfo=UTC), success=True, status_code=200, html=html)
    return normalize(page, source, CrawlLimits(**limits))


def test_article_missing_optional_fields_and_navigation():
    html = "<html><nav>Navigation must disappear</nav><article><p>" + (
        "The delegation discussed peace negotiations with the council. " * 4
    ) + "</p></article></html>"
    item = normalized(html)
    assert item.author is None and item.title is None and item.published_at is None
    assert "Navigation" not in item.body
    assert "Author unavailable" in item.metadata["warnings"]


def test_article_structured_metadata_and_invalid_date():
    item = normalized('''<script type="application/ld+json">{"@type":"NewsArticle",
        "headline":"Talks", "author":{"name":"Reporter"}, "datePublished":"not a date"}</script>
        <article><p>''' + "Officials discussed peace and negotiations. " * 5 + "</p></article>")
    assert item.title == "Talks" and item.author == "Reporter"
    assert item.published_at is None
    assert any("Unparsed" in warning for warning in item.metadata["warnings"])


def test_reddit_comments_keep_authors_and_parents():
    item = normalized('''<shreddit-post id="t3_abc" post-title="Diplomacy" author="poster">
        <div slot="text-body">Post text</div></shreddit-post>
        <shreddit-comment thingid="t1_a" author="alice" parentid="t3_abc">
          <div slot="comment">First comment about diplomacy.</div>
          <shreddit-comment thingid="t1_b" author="bob" parentid="t1_a">
            <div slot="comment">Reply about sanctions.</div>
          </shreddit-comment>
        </shreddit-comment>''', "reddit")
    assert [s.id for s in item.segments] == ["t3_abc", "t1_a", "t1_b"]
    assert item.segments[1].text == "First comment about diplomacy."
    assert item.segments[2].parent_id == "t1_a"
    assert item.segments[2].author == "bob"


def test_x_short_post_is_not_dropped_or_replaced_by_another_post():
    html = '''<article data-testid="tweet"><a href="/user/status/2">Other</a>
    <div data-testid="tweetText">Unrelated post</div></article>
    <article data-testid="tweet"><a href="/user/status/1">Target</a>
    <div data-testid="User-Name">Public institution @institution</div>
    <div data-testid="tweetText">Peace talks resumed.</div>
    <time datetime="2026-10-07T10:00:00Z"></time></article>'''
    item = normalized(html, "x", "https://example.com/user/status/1")
    assert item.body == "Peace talks resumed."
    assert item.title is None
    assert item.published_at.hour == 10


@pytest.mark.parametrize("adapter", ["article", "reddit", "x"])
def test_blocked_page_is_not_success(adapter):
    with pytest.raises(NormalizationError, match="access_blocked"):
        normalized("<title>Access Denied</title><body>Please log in</body>", adapter)


def test_size_cap_and_content_hash_are_stable():
    item = normalized("<article>" + "Diplomacy and sanctions. " * 100 + "</article>", max_body_characters=200)
    assert len(item.body) == 200
    assert item.metadata["truncated"]
    later = item.model_copy(update={"scraped_at": datetime(2026, 10, 8, tzinfo=UTC)})
    assert item.content_hash == later.content_hash


HN_THREAD = """<html><table>
<tr class="athing submission" id="100"><td><span class="titleline"><a href="https://news.example/iran">
UN Security Council debates Iran sanctions</a></span></td></tr>
<tr><td class="subtext"><span class="subline"><a class="hnuser" href="user?id=poster">poster</a>
<span class="age" title="2026-03-12T10:00:00 1773309600"><a href="item?id=100">1 day ago</a></span></span></td></tr>
<tr class="athing comtr" id="101"><td><td class="ind" indent="0"></td><span class="comhead">
<a class="hnuser">alice</a><span class="age" title="2026-03-12T11:00:00"></span></span>
<div class="comment"><div class="commtext c00">Russia and China dispute the snapback mechanism.</div>
<div class="reply"><a href="reply?id=101">reply</a></div></div></td></tr>
<tr class="athing comtr" id="102"><td><td class="ind" indent="1"></td><span class="comhead">
<a class="hnuser">bob</a></span><div class="comment"><div class="commtext c00">France supported the vote.</div></div></td></tr>
<tr class="athing comtr" id="103"><td><td class="ind" indent="2"></td><div class="comment"><span>[flagged]</span></div></td></tr>
<tr class="athing comtr" id="104"><td><td class="ind" indent="3"></td><span class="comhead"><a class="hnuser">carol</a></span>
<div class="comment"><div class="commtext c00">Reply to a flagged comment keeps its real parent.</div></div></td></tr>
<tr class="athing comtr" id="105"><td><td class="ind" indent="0"></td><span class="comhead"><a class="hnuser">dave</a></span>
<div class="comment"><div class="commtext c00">A second top-level comment.</div></div></td></tr>
</table></html>"""


def test_hackernews_thread_keeps_reply_structure_and_utc_times():
    """Synthetic markup mirroring the live Hacker News item page structure."""
    item = normalized(HN_THREAD, adapter="hackernews", url="https://news.ycombinator.com/item?id=100")
    assert item.title == "UN Security Council debates Iran sanctions" and item.author == "poster"
    assert item.published_at == datetime(2026, 3, 12, 10, 0, tzinfo=UTC)
    parents = {s.id: s.parent_id for s in item.segments}
    assert parents == {"100": None, "101": "100", "102": "101", "104": "103", "105": "100"}
    by_id = {s.id: s for s in item.segments}
    assert by_id["101"].text == "Russia and China dispute the snapback mechanism."  # reply link removed
    assert by_id["101"].published_at == datetime(2026, 3, 12, 11, 0, tzinfo=UTC)
    assert by_id["102"].url == "https://news.ycombinator.com/item?id=102"
    assert any("external link" in w for w in item.metadata["warnings"])


def test_hackernews_comment_cap_and_missing_story():
    item = normalized(HN_THREAD, adapter="hackernews", url="https://news.ycombinator.com/item?id=100",
                      max_comments_per_thread=2)
    assert len(item.segments) == 3 and item.metadata["truncated"]
    with pytest.raises(NormalizationError, match="unsupported_structure"):
        normalized("<html><p>Comment permalink</p></html>", adapter="hackernews",
                   url="https://news.ycombinator.com/item?id=101")


def test_listing_page_is_followed_but_never_stored():
    """Synthetic section page: accepted as a seed, rejected as a content item."""
    source = SourceProfile(name="bbc", source_type="news", adapter="article", allowed_domains=["www.bbc.com"],
                           seeds=["https://www.bbc.com/news/world/asia/india"],
                           include_paths=["^/news/articles/[a-z0-9]+$"], listing_paths=["^/news/world/asia/india$"])
    assert source.accepts("https://www.bbc.com/news/articles/c5j9kngz3j1go")
    assert not source.accepts("https://www.bbc.com/sport/cricket")
    page = FetchResult(requested_url=source.seeds[0], final_url=source.seeds[0], source_name="bbc", depth=0,
                       success=True, status_code=200, html="<main>" + "Headline link text. " * 20 + "</main>")
    with pytest.raises(NormalizationError, match="listing_page"):
        normalize(page, source, CrawlLimits())


MASTODON_THREAD = """<html><head><meta property="og:published_time" content="2026-09-30T10:00:00Z"></head><body>
<div class="status status-reply" data-id="90"><span class="display-name__account">@earlier@mastodon.social</span>
<div class="status__content__text"><p>Context post before the focused one.</p></div></div>
<div class="detailed-status"><span class="display-name__account">@arstechnica@mastodon.social</span>
<div class="status__content__text"><p>OpenAI delays IPO over AI safety concerns</p><p>Read more:<br>
<a class="unhandled-link" href="https://arstechnica.com/ai/openai-ipo/?utm_source=mastodon"><span class="invisible">https://</span>
<span class="ellipsis">arstechnica.com/ai/</span><span class="invisible">openai-ipo/</span></a></p></div></div>
<div class="status status-reply" data-id="101"><span class="display-name__account">@reader@infosec.exchange</span>
<a class="status__relative-time" href="/@reader@infosec.exchange/101"><time datetime="2026-09-30T11:00:00.000Z">1d</time></a>
<div class="status__content__text"><p><span class="h-card"><a class="mention u-url" href="/@arstechnica">@<span>arstechnica</span></a></span>
Sam Altman keeps moving the goalposts.</p></div></div>
<div class="status status-reply" data-id="102"><div class="status__content__text"><p></p></div></div>
</body></html>"""


def test_mastodon_thread_keeps_mentions_inline_and_context_unattached():
    """Synthetic markup mirroring a rendered mastodon.social thread."""
    item = normalized(MASTODON_THREAD, adapter="mastodon", url="https://mastodon.social/@arstechnica/100")
    assert item.author == "@arstechnica@mastodon.social" and item.title is None
    assert item.published_at == datetime(2026, 9, 30, 10, 0, tzinfo=UTC)
    by_id = {s.id: s for s in item.segments}
    assert list(by_id) == ["90", "100", "101"]  # empty reply skipped
    assert by_id["100"].text == "OpenAI delays IPO over AI safety concerns\nRead more:\nhttps://arstechnica.com/ai/openai-ipo/"
    assert by_id["101"].text == "@arstechnica Sam Altman keeps moving the goalposts."
    assert (by_id["101"].parent_id, by_id["101"].kind) == ("100", "reply")
    assert (by_id["90"].parent_id, by_id["90"].kind) == (None, "context")
    assert by_id["101"].url == "https://mastodon.social/@reader@infosec.exchange/101"


def test_mastodon_unrendered_shell_is_not_content():
    with pytest.raises(NormalizationError, match="access_or_structure"):
        normalized("<html><body><div id='mastodon'></div></body></html>", adapter="mastodon",
                   url="https://mastodon.social/@arstechnica/100")


def test_exclude_selectors_drop_related_stories_and_signup_forms():
    """Shaped like the Al Jazeera body (synthetic text): a related-stories list and a newsletter form sit inside the
    body selector; their headlines must not be credited to this article."""
    html = """<div class="wysiwyg"><p>Officials in Geneva discussed the sanctions regime with envoys on Monday.</p>
      <section class="more-on"><h2>Recommended Stories</h2><ul><li>list 1 of 2US issues new Iran sanctions</li>
      <li>list 2 of 2Other headline about Hormuz</li></ul></section>
      <div class="sib-newsletter-form"><h4>Sign up for Al Jazeera</h4><span>protected by reCAPTCHA</span></div>
      <div class="container--ads">Advertisement</div>
      <p>The envoys said the talks would continue next week in the Swiss city, according to two diplomats.</p></div>"""
    source = SourceProfile(name="test", source_type="news", adapter="article", seeds=["https://example.com/story"],
                           allowed_domains=["example.com"], body_selector=".wysiwyg",
                           exclude_selectors=[".more-on", ".sib-newsletter-form", ".container--ads"])
    page = FetchResult(requested_url="https://example.com/story", final_url="https://example.com/story",
                       source_name="test", depth=0, fetched_at=datetime(2026, 10, 7, tzinfo=UTC), success=True,
                       status_code=200, html=html)
    body = normalize(page, source, CrawlLimits()).body
    assert body.startswith("Officials in Geneva") and body.endswith("according to two diplomats.")
    for boilerplate in ("Recommended Stories", "list 1 of 2", "Sign up", "reCAPTCHA", "Advertisement"):
        assert boilerplate not in body
    # Without the option the same blocks stay (the shipped config sets it for aljazeera only).
    plain = source.model_copy(update={"exclude_selectors": []})
    assert "Recommended Stories" in normalize(page, plain, CrawlLimits()).body


def test_embedded_recommendation_list_is_removed_from_article_body():
    """Synthetic body modelled on the live Al Jazeera recommendation widget text."""
    paragraph = "The US military said the strait remained open while talks with Iran continued in Oman. " * 3
    html = ("<article><p>" + paragraph + "</p><ul><li>Recommended Stories</li><li>list of 2 items</li>"
            "<li>list 1 of 2</li><li>Has Iran brought down a US F-35 jet?</li><li>list 2 of 2</li>"
            "<li>Ethiopia, Eritrea break ties</li><li>end of list</li></ul><p>" + paragraph + "</p></article>")
    item = normalized(html)
    assert "Eritrea" not in item.body and "list of 2 items" not in item.body
    assert item.body.count("talks with Iran") == 6
    assert "Removed embedded recommendation list(s) from the article body" in item.metadata["warnings"]
