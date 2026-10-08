"""Source adapters retain evidence segments instead of mixing unrelated comments."""

import json
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, NavigableString, Tag

from .config import CrawlLimits, SourceProfile, canonical_url
from .models import ContentItem, FetchResult, Segment

BLOCK_TAGS = ["address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure", "h1", "h2",
              "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p", "pre", "section", "table", "td", "th", "tr", "ul"]
REDDIT_THREAD = re.compile(r"^/r/([^/]+)/comments/([a-z0-9]+)", re.IGNORECASE)
# Screen-reader text of a recommendation widget inside an article body ("list of 4 items ... end of list").
# Its headlines belong to other articles; left in, they become entities and edges credited to this one.
EMBEDDED_LIST = re.compile(r"^(?:Recommended Stories\n)?list of \d+ items\n[\s\S]*?^end of list$\n?", re.MULTILINE)
MAX_TITLE_CHARS = 500
REDDIT_CANONICAL_HOST = "www.reddit.com"


class NormalizationError(ValueError):
    """An explicit skipped item, rather than a silently empty successful source."""


def clean_text(value: str) -> str:
    return "\n".join(line for line in (re.sub(r"[ \t\r\f\v]+", " ", s).strip()
                                      for s in value.splitlines()) if line)


def text_of(element: Tag | None) -> str:
    """Visible text with a line break only between blocks, so inline links stay inside their sentence and
    extraction can treat every line break as a sentence boundary."""
    if element is None:
        return ""
    clone = BeautifulSoup(str(element), "html.parser")
    for tag in clone.select("script, style, nav, footer, header, button, form, aside, noscript"):
        tag.decompose()
    for text in clone.find_all(string=True):
        if type(text) is NavigableString and text.find_parent("pre") is None:
            text.replace_with(re.sub(r"\s+", " ", text))  # HTML renders any whitespace run as one space
    for tag in clone.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    return clean_text(clone.get_text())


def parse_date(value: str | None, warnings: list[str]) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (ValueError, TypeError, IndexError):
            warnings.append(f"Unparsed publication time: {value[:80]}")
            return None
    if parsed.tzinfo is None:
        warnings.append("Publication time has no timezone; retained as raw metadata")
        return None
    return parsed.astimezone(UTC)


def meta(soup: BeautifulSoup, *keys: str) -> str | None:
    for key in keys:
        element = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if element and element.get("content"):
            return str(element["content"]).strip() or None
    return None


def structured_objects(soup: BeautifulSoup) -> list[dict]:
    result = []

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            result.append(value)
            if "@graph" in value:
                visit(value["@graph"])

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            visit(json.loads(script.get_text()[:2_000_000]))
        except (ValueError, TypeError):
            continue
    return result


def author_name(value) -> str | None:
    if isinstance(value, dict):
        return value.get("name")
    if isinstance(value, list):
        names = [author_name(v) for v in value]
        return ", ".join(n for n in names if n) or None
    return value.strip() if isinstance(value, str) and value.strip() else None


def blocked_page(soup: BeautifulSoup) -> str | None:
    title = text_of(soup.title).lower()
    body = text_of(soup.body or soup)[:2500].lower()
    markers = ("you've been blocked by network security", "blocked by network security",
               "verify you are human", "checking your browser", "access denied",
               "just a moment...", "enable javascript and cookies to continue")
    if any(marker in title for marker in markers):
        return "access_blocked: challenge or denial page"
    if len(body) < 2000 and any(marker in body for marker in markers):
        return "access_blocked: challenge or denial page"
    return None


def _article(soup, source, warnings, base_url):
    objects = structured_objects(soup)
    article = next((o for o in objects if any(t in str(o.get("@type", ""))
                    for t in ("NewsArticle", "Article", "BlogPosting"))), {})
    title = article.get("headline") or meta(soup, "og:title", "twitter:title") or text_of(soup.h1) or None
    author = author_name(article.get("author")) or meta(soup, "author", "article:author")
    raw_date = article.get("datePublished") or meta(soup, "article:published_time", "datePublished")
    if not raw_date:
        time_element = soup.find("time", datetime=True)
        raw_date = time_element.get("datetime") if time_element else None
    container = soup.select_one(source.body_selector) if source.body_selector else None
    if source.body_selector and container is None:
        warnings.append("Configured body selector did not match; tried article/main metadata fallback")
    if container is None:
        container = soup.select_one('[itemprop="articleBody"], article, main')
    if container is not None and source.exclude_selectors:
        # Recommended-story lists and sign-up forms sit inside the article body on some sites; their headlines would
        # otherwise become entities and edges credited to this article.
        for element in container.select(", ".join(source.exclude_selectors)):
            element.decompose()
    body = text_of(container) or clean_text(str(article.get("articleBody", "")))
    body, removed = EMBEDDED_LIST.subn("", body)
    if removed:
        warnings.append("Removed embedded recommendation list(s) from the article body")
    if not body or len(body) < 120:
        raise NormalizationError("empty_content: no substantial article body; listing/login pages are not articles")
    return title, author, raw_date, [Segment(id="body", text=body, author=author, url=base_url)]


def _reddit(soup, source, warnings, base_url):
    post = soup.find("shreddit-post") or soup.select_one(".thing.link")
    if post is None:
        raise NormalizationError("unsupported_structure: no Reddit post; page may require login or be a listing")
    title = post.get("post-title") or text_of(soup.h1) or text_of(post.select_one("a.title")) or None
    author = post.get("author") or text_of(post.select_one(".author")) or None
    if author in {"[deleted]", "[removed]"}:
        author = None
    root_id = post.get("id") or post.get("data-fullname") or "post"
    raw_date = post.get("created-timestamp")
    time_element = post.select_one("time[datetime]")
    if not raw_date and time_element:
        raw_date = time_element.get("datetime")
    post_body = post.select_one('[slot="text-body"], .usertext-body .md, .md')
    root_text = text_of(post_body) or title or ""
    segments = [Segment(id=root_id, text=root_text, kind="post", author=author, url=base_url)] if root_text else []
    comments = soup.select("shreddit-comment") or soup.select(".thing.comment")
    for index, comment in enumerate(comments):
        cid = comment.get("thingid") or comment.get("id") or comment.get("data-fullname") or f"comment-{index}"
        body = comment.select_one('[slot="comment"], .entry .usertext-body .md, .usertext-body .md')
        # Do not let a deleted parent's descendant comment become its own body.
        if body is None or body.find_parent("shreddit-comment") not in (None, comment):
            continue
        old_parent = body.find_parent(class_=lambda c: c and "comment" in c.split())
        if comment.name != "shreddit-comment" and old_parent is not None and old_parent is not comment:
            continue
        text = text_of(body)
        if text in {"", "[deleted]", "[removed]"}:
            continue
        cauthor = comment.get("author") or text_of(comment.select_one(".author")) or None
        if cauthor in {"[deleted]", "[removed]"}:
            cauthor = None
        ancestor = comment.find_parent("shreddit-comment") or comment.find_parent(class_="comment")
        parent_id = comment.get("parentid") or (ancestor.get("thingid") or ancestor.get("id") if ancestor else root_id)
        permalink = comment.get("permalink")
        if not permalink:
            link = comment.select_one("a.bylink")
            permalink = link.get("href") if link else None
        comment_time = comment.get("created") or comment.get("created-timestamp")
        time_element = comment.select_one("time[datetime]")
        if not comment_time and time_element:
            comment_time = time_element.get("datetime")
        segments.append(Segment(id=str(cid), text=text, kind="comment", author=cauthor,
                                parent_id=parent_id, url=urljoin(base_url, permalink) if permalink else None,
                                published_at=parse_date(comment_time, warnings)))
    if len(segments) <= 1:
        warnings.append("No visible comments were extracted; thread coverage is partial")
    if not segments:
        raise NormalizationError("empty_content: no Reddit post or comment text")
    return title, author, raw_date, segments


def _x(soup, source, warnings, base_url):
    post_id = urlsplit(base_url).path.rstrip("/").split("/")[-1]
    tweets = soup.select('article[data-testid="tweet"]')
    tweet = next((t for t in tweets if any(
        re.search(rf"/status/{re.escape(post_id)}(?:$|[/?])", a.get("href", ""))
        for a in t.select("a[href]")
    )), None)
    if tweet is None:
        raise NormalizationError("access_or_structure: target X post unavailable; login/shell pages do not count")
    text_element = tweet.select_one('[data-testid="tweetText"]')
    body = text_of(text_element)
    if not body:
        raise NormalizationError("empty_content: target X post contains no visible text")
    user = tweet.select_one('[data-testid="User-Name"]')
    author = text_of(user) or None
    date_element = tweet.select_one("time[datetime]")
    raw_date = date_element.get("datetime") if date_element else None
    warnings.append("Only the requested visible X post is normalized; hidden replies are not claimed")
    return None, author, raw_date, [Segment(id=post_id, text=body, kind="post", author=author, url=base_url)]


def _hn_time(element, warnings) -> datetime | None:
    # HN's age title is "YYYY-MM-DDTHH:MM:SS[ unixtime]"; HN documents these times as UTC.
    raw = (element.get("title") or "").split() if element else []
    if len(raw) > 1 and raw[1].isdigit():
        return datetime.fromtimestamp(int(raw[1]), UTC)
    if raw:
        try:
            return datetime.fromisoformat(raw[0]).replace(tzinfo=UTC)
        except ValueError:
            warnings.append(f"Unparsed Hacker News time: {raw[0][:40]}")
    return None


def _hackernews(soup, source, warnings, base_url):
    story_id = dict(parse_qsl(urlsplit(base_url).query)).get("id", "story")
    # Only a story row has a title line; a comment permalink's top row (also tr.athing) is not a thread.
    story = soup.select_one("tr.athing.submission") or next(
        (row for row in soup.select("tr.athing") if row.get("id") == story_id and row.select_one(".titleline")), None)
    if story is None:
        raise NormalizationError("unsupported_structure: no Hacker News story; comment permalinks are not items")
    link = story.select_one(".titleline > a")
    title = text_of(link) or None
    subline = soup.select_one(".subline") or soup.select_one(".subtext")
    author = text_of(subline.select_one("a.hnuser")) if subline else None
    story_time = _hn_time(subline.select_one(".age") if subline else None, warnings)
    toptext = text_of(soup.select_one(".toptext"))
    root_text = "\n".join(part for part in (title, toptext) if part)
    segments = [Segment(id=story_id, text=root_text, kind="post", author=author or None, url=base_url,
                        published_at=story_time)] if root_text else []
    # Reply structure is encoded only by indentation; keep a stack of the latest comment at each level.
    stack: list[str] = []
    for row in soup.select("tr.athing.comtr"):
        cid = row.get("id") or f"comment-{len(stack)}"
        indent_cell = row.select_one("td.ind")
        level = int(indent_cell.get("indent", 0)) if indent_cell else 0
        parent_id = stack[level - 1] if 0 < level <= len(stack) else story_id
        del stack[level:]
        stack.append(cid)
        body = row.select_one(".commtext")
        if body is None:
            continue
        for reply in body.select(".reply"):
            reply.decompose()
        text = text_of(body)
        if text in {"", "[deleted]", "[flagged]", "[dead]"}:
            continue
        cauthor = text_of(row.select_one(".comhead a.hnuser")) or None
        segments.append(Segment(id=cid, text=text, kind="comment", author=cauthor, parent_id=parent_id,
                                url=urljoin(base_url, f"item?id={cid}"),
                                published_at=_hn_time(row.select_one(".comhead .age"), warnings)))
    if len(segments) <= 1:
        warnings.append("No visible comments were extracted; thread coverage is partial")
    if not segments:
        raise NormalizationError("empty_content: no Hacker News story or comment text")
    if link is not None and link.get("href"):
        warnings.append(f"Discussion of external link (not crawled): {urljoin(base_url, link['href'])[:200]}")
    raw_date = story_time.isoformat() if story_time else None
    return title, author or None, raw_date, segments


def _mastodon_text(element: Tag | None) -> str:
    # Mentions nest "@" and the name in separate spans and links hide URL parts, so the generic newline-joined
    # text would split "@user" and URLs apart; rebuild each paragraph inline instead.
    if element is None:
        return ""
    clone = BeautifulSoup(str(element), "html.parser")
    for hidden in clone.select(".invisible"):
        hidden.unwrap() if hidden.find_parent("a", class_="mention") else hidden.decompose()
    for link in clone.select("a[href]"):
        classes = link.get("class") or []
        if "mention" not in classes and "hashtag" not in classes:
            link.replace_with(link.get("href", "").split("?")[0])
    for br in clone.find_all("br"):
        br.replace_with("\x00")  # explicit line break; other whitespace collapses as in rendered HTML
    paragraphs = clone.find_all("p") or [clone]
    return clean_text("\n".join(" ".join(p.get_text("").split()).replace("\x00", "\n") for p in paragraphs))


def _mastodon(soup, source, warnings, base_url):
    post_id = urlsplit(base_url).path.rstrip("/").split("/")[-1]
    focused = soup.select_one(".detailed-status")
    if focused is None:
        raise NormalizationError("access_or_structure: Mastodon post did not render; profile/shell pages do not count")
    def account(node):
        handle = text_of(node.select_one(".display-name__account")) if node else ""
        return handle or None
    author = account(focused)
    body = _mastodon_text(focused.select_one(".status__content__text"))
    if not body:
        raise NormalizationError("empty_content: Mastodon post contains no visible text")
    raw_date = meta(soup, "og:published_time")
    segments, before_focus = [], True
    for node in soup.select(".detailed-status, .status[data-id]"):
        if node is focused:
            before_focus = False
            segments.append(Segment(id=post_id, text=body, kind="post", author=author, url=base_url,
                                    published_at=parse_date(raw_date, warnings)))
            continue
        text = _mastodon_text(node.select_one(".status__content__text"))
        if not text:
            continue
        link = node.select_one("a.status__relative-time[href]")
        stamp = node.select_one("time[datetime]")
        # The rendered thread does not expose reply targets; replies attach to the focused post, earlier
        # context posts stay unattached rather than inventing a reply chain.
        segments.append(Segment(id=str(node["data-id"]), text=text, kind="context" if before_focus else "reply",
                                author=account(node), parent_id=None if before_focus else post_id,
                                url=urljoin(base_url, link["href"]) if link else None,
                                published_at=parse_date(stamp.get("datetime") if stamp else None, warnings)))
    if len(segments) <= 1:
        warnings.append("No visible replies were extracted; thread coverage is partial")
    warnings.append("Mastodon reply targets are not exposed in rendered HTML; replies are attached to the post")
    return None, author, raw_date, segments


def _forum(soup, source, warnings, base_url):
    # Configurable community adapter for later approved sources; never used as an automatic fallback.
    title = text_of(soup.h1) or None
    selector = source.body_selector or ".s-prose"
    parts = [text_of(node) for node in soup.select(selector)]
    parts = [part for part in parts if part]
    if not parts:
        raise NormalizationError("unsupported_structure: no community content matched")
    warnings.append("Generic forum adapter: author/reply relationships may be unavailable")
    return title, None, None, [Segment(id=f"segment-{i}", text=p, kind="discussion") for i, p in enumerate(parts)]


def thread_url(adapter: str, url: str) -> str:
    """The logical item URL (07): every permalink, sort order, or host variant of one discussion thread maps to the
    thread, so revisiting it through another URL can never count as a separate source."""
    parts = urlsplit(url)
    if adapter == "reddit" and (match := REDDIT_THREAD.match(parts.path)):
        # old./new./np. host variants of one thread share one identity; this is a canonical form, not a crawl target.
        return urlunsplit(("https", REDDIT_CANONICAL_HOST, f"/r/{match[1].lower()}/comments/{match[2].lower()}/", "", ""))
    if adapter == "hackernews" and (story := dict(parse_qsl(parts.query)).get("id")):
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode({"id": story}), ""))
    return url


def headline(title: str | None) -> str:
    """The title as one line without a trailing " | Site name" suffix."""
    text = " ".join((title or "").split())
    return (text.rsplit(" | ", 1)[0] if " | " in text else text)[:MAX_TITLE_CHARS]


def normalize(page: FetchResult, source: SourceProfile, limits: CrawlLimits) -> ContentItem:
    if not page.success or (page.status_code is not None and page.status_code >= 400):
        raise NormalizationError(page.error or f"http_error: {page.status_code}")
    soup = BeautifulSoup(page.html, "html.parser")
    blocked = blocked_page(soup)
    if blocked:
        raise NormalizationError(blocked)
    if source.is_listing(page.final_url):
        raise NormalizationError("listing_page: navigation-only page; links followed, not stored as content")
    warnings: list[str] = []
    adapters = {"article": _article, "reddit": _reddit, "x": _x, "forum": _forum, "hackernews": _hackernews,
                "mastodon": _mastodon}
    fetched = canonical_url(page.final_url)
    url = thread_url(source.adapter, fetched)
    title, author, raw_date, segments = adapters[source.adapter](soup, source, warnings, url)
    published_at = parse_date(raw_date, warnings)
    truncated = False
    if source.adapter in {"reddit", "hackernews", "mastodon"} and len(segments) > limits.max_comments_per_thread + 1:
        segments = segments[:limits.max_comments_per_thread + 1]
        warnings.append("Comment count capped by configuration")
        truncated = True
    remaining = limits.max_body_characters
    kept = []
    for segment in segments:
        if len(segment.text) > remaining:
            truncated = True
            segment = segment.model_copy(update={"text": segment.text[:remaining]})
        if segment.text:
            kept.append(segment)
            remaining -= len(segment.text) + 2
        if remaining <= 0:
            break
    if truncated:
        warnings.append("Content was truncated by the configured size/comment limits")
    body = "\n\n".join(s.text for s in kept)
    if not body.strip():
        raise NormalizationError("empty_content: all extracted segments were empty")
    if author is None:
        warnings.append("Author unavailable")
    if published_at is None:
        warnings.append("Publication timestamp unavailable")
    # An article headline is extracted like the body (E17), as its own segment so evidence offsets stay exact; the
    # body field keeps only the article text. Thread titles of other adapters already open their post segment.
    if source.adapter == "article" and (line := headline(title)) and not any(line in s.text for s in kept):
        kept.insert(0, Segment(id="title", text=line, kind="title", url=url))
    metadata = {"source_name": source.name, "adapter": source.adapter, "depth": page.depth, "warnings": warnings,
                "truncated": truncated, "published_at_raw": raw_date}
    if fetched != url:
        metadata["fetched_url"] = fetched
    return ContentItem(source_url=url, source_type=source.source_type, scraped_at=page.fetched_at,
                       requested_url=page.requested_url, title=title, body=body, author=author,
                       published_at=published_at, segments=kept, metadata=metadata)
