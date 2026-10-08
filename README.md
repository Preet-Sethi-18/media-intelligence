# Media Intelligence — evidence-backed knowledge graph

Crawls three different kinds of public sources with **Crawl4AI**, normalises them into one schema, extracts
people, organisations, locations and topics plus **typed relationships** with spaCy and explicit rules, stores
the graph in **SQLite** with a citation for every edge, and serves three analysis endpoints with **FastAPI**.

```
config/sources.toml ──► crawler.py (Crawl4AI, BFS frontier, whitelist, robots.txt, per-host delay)
                         │
                         ▼
                   normalization.py (one adapter per page structure → ContentItem + segments)
                         │
                         ▼
                   extraction.py (spaCy NER + aliases.toml + topics.toml + dependency rules)
                         │
                         ▼
                   storage.py ─► SQLite (nodes, edges, sources, edge_sources, …)   ◄── pipeline.py / cli.py
                                         │
                                         ▼
                   analysis.py ─► api.py (FastAPI): /entity/{name}/network, /connections/new, /entities/central
```

## Sources (three distinct source types)

| Source | `source_type` | Structure | Topic |
| --- | --- | --- | --- |
| Al Jazeera articles | `news` | One article body (JSON-LD/meta metadata) | Geopolitics (Iran, sanctions, conflict; US–China AI and chip policy) |
| Hacker News threads | `discussion` | Story + comment tree (reply parents from indentation) | AI industry (models, deals, CEOs) |
| Mastodon posts (mastodon.social) | `microblog` | Post + replies (client-rendered) | AI industry, from tech outlets' accounts |

Hacker News and Mastodon cover the same AI-industry stories from two angles (a community discussion thread vs.
a publisher's post with public replies). Al Jazeera supplies the news source; its AI-policy seeds (the US export
ban on Anthropic's models, Nvidia/Supermicro staff charged over AI-server exports) put the same companies in a
geopolitical frame, so `Anthropic`, `Nvidia` and `OpenAI` connect across all three source types.

**Why not Reddit/X:** both were the original plan, but a live Crawl4AI probe was refused by their robots.txt
(`User-agent: * / Disallow: /`). This project respects robots.txt and does not use anti-bot evasion, so they
were replaced. Other candidates were tested live and rejected: UN press releases (bot wall, HTTP 406), IAEA and
Politics Stack Exchange (Cloudflare challenge), Quora (robots.txt).

## Quick start

Requires Linux/macOS, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <URL of this repository> media-intelligence && cd media-intelligence
uv sync --locked                       # dependencies + pinned spaCy model en_core_web_sm 3.8.0
uv run crawl4ai-setup                  # Crawl4AI's browser setup
uv run python -m playwright install --with-deps chromium   # browser + OS libraries (needs sudo for the OS part)
uv run crawl4ai-doctor                 # must print "Crawling test passed"

uv run media-intelligence validate-config            # checks config/sources.toml without crawling
uv run media-intelligence ingest                     # crawl → normalise → extract → SQLite + run report
uv run uvicorn media_intelligence.api:app --port 8000

curl 'http://127.0.0.1:8000/entities/central?limit=10'
curl 'http://127.0.0.1:8000/entity/OpenAI/network?depth=2'
# Growth needs two ingests: after one, every edge is "new". Ingest again later (or with changed seeds) and
# use the second run's started_at from its reports/run-*/report.json; a future `since` returns 422.
curl 'http://127.0.0.1:8000/connections/new?since=<started_at of the second run>'
uv run pytest -q                                     # offline tests, no network needed
```

Settings come from environment variables or `.env` (see `.env.example`): `MI_DATABASE_PATH`
(default `data/media_intelligence.sqlite3`), `MI_REPORT_DIR`, `MI_SPACY_MODEL`, `MI_GROWTH_MIN_DELTA`,
`MI_GROWTH_MIN_RATIO`. Exit codes: `0` complete, `1` fatal, `2` setup/config error, `3` partial
(a configured source stored nothing, or a seed failed).

## Configuration and changing the seed list

Everything crawl-related is in [config/sources.toml](config/sources.toml); no crawl URL or seed domain is
hard-coded in `src/` (the only domain literal is the canonical Reddit host used to merge thread URL variants for
the unused `reddit` adapter). Global limits: `max_depth`, `max_pages_total`, `max_pages_per_source`,
timeouts, retries, `delay_per_host_seconds`, `respect_robots`. Each `[[sources]]` block has:

- `seeds` — start URLs; `allowed_domains` — exact-host whitelist; `include_paths` — regexes for content pages.
- `adapter` — which page structure to normalise: `article`, `hackernews`, `mastodon` (also `reddit`, `x`,
  `forum`, unused here).
- Optional per-source overrides that can only tighten global limits: `max_depth`, `min_delay_seconds`
  (Hacker News asks for `Crawl-delay: 30`), `listing_paths` (index pages used as entry points, never stored),
  `follow_links = "listings"` (follow links only from listing pages).

**To swap seeds:** replace the URLs in `seeds` and rerun `ingest`. To keep the shipped file, copy it and run
`uv run media-intelligence ingest --config my/sources.toml`; `aliases.toml` and `topics.toml` are read from
the same directory, so copy them too. Valid seeds include any dated Al Jazeera article under `/news/`,
`/features/` or `/economy/` (other sections such as `/opinions/` need a wider `include_paths`) or a listed
section/tag page, a Hacker News thread (`/item?id=…`) or listing (`/`, `/best`, `/newest`, `/ask`, `/show`),
and any mastodon.social post URL. Verified live with no code changes: run 3 below (all 13 seeds replaced) and
an earlier development run on a separate scratch database whose only Hacker News seed was the front page
(both exit 0); `tests/unit/test_config.py::test_every_seed_swapped_needs_no_code_change` covers it offline. A
seed outside every source's whitelist is rejected by `validate-config` with an explicit error; to crawl a new
site, add a `[[sources]]` block (the `article` adapter handles most news sites via JSON-LD, `<article>` and
`<main>`).

Crawl boundaries: breadth-first with depth counted from the seeds, exact-host whitelist checked on every link
and after redirects, private/local addresses refused, robots.txt checked by Crawl4AI, browser navigations
outside the whitelist aborted.

## Normalised schema

Every page becomes a `ContentItem` ([models.py](src/media_intelligence/models.py)): `source_url`, `source_type`,
`scraped_at`, `title`, `body`, `author`, `published_at`, plus `segments` (the post and each comment or reply
kept separately, with its own author, time, parent and permalink) and `metadata` (warnings, truncation, raw
date). Missing fields are `null` and counted in the run report's `missing_metadata` (for example, Mastodon
posts have no title); a page with no usable body is recorded as a skip with a reason, never stored as empty
content. Login walls, challenge pages and comment permalinks are rejected explicitly.

## Entities and normalisation

Final database: 558 nodes (239 ORG, 165 PERSON, 138 LOCATION, 16 TOPIC) and 794 alias rows (647 observed
surface forms plus 147 reviewed aliases/handles).

- Types: `PERSON`, `ORG`, `LOCATION` (spaCy `GPE`/`LOC`/`FAC`), `TOPIC` (whole-token phrase patterns in
  [config/topics.toml](config/topics.toml)).
- **"Elon Musk" / "Musk" / "@elonmusk":** resolution order per item: (1) exact canonical name; (2) a reviewed
  alias or handle from [config/aliases.toml](config/aliases.toml) (`@elonmusk` → `Elon Musk`, `UN` →
  `United Nations`); (3) a bare surname or first name resolves to a full name only if exactly one full name in
  the same item ends (or starts) with it; two candidates leave it unresolved instead of guessing. Item context
  also beats a reviewed single-word alias: "Musk" next to "Kimbal Musk" is left unresolved, while a lone "Musk"
  maps to Elon Musk. Matching is exact after Unicode/case/whitespace normalisation (person titles and a leading
  "the" are trimmed); there is no fuzzy matching. Unmapped `@handles` and usernames never become person nodes.
  Identity is type-aware (`PERSON:elon musk` ≠ `ORG:…`). Regression test: `tests/unit/test_extraction.py`.
- Where it fails on our real data (split first names, a company typed two ways, capital vs. country, products
  typed as people): see [hard question 2](#2-how-entity-normalisation-breaks).
- Every observed surface form is stored in `node_aliases`, so merges can be inspected, and the API resolves
  names through it (an ambiguous name returns HTTP 409 with candidates).

## Relationship types

Defined once in `extraction.RELATION_DEFINITIONS` (rules version `rules-5`); each rule function's docstring
lists its triggers, suppressions and known failures. Counts are distinct edges in the final database.

| Relation | Direction | Meaning and how it is detected | Edges |
| --- | --- | --- | --- |
| `affiliated_with` | A → ORG B | Current role in, works for, leads/runs or founded B: titles ("OpenAI CEO Sam Altman"), appositives ("Esmaeil Baghaei, the Iranian Foreign Ministry spokesman"), "B's A", "A of B", lead/run/found/work-for verbs. Former roles excluded. | 18 |
| `criticized` | A → B | A criticized, condemned, accused or blamed B (or B's action). The PDF's `accused_of` is folded in. | 10 |
| `sanctioned` | A → B | A imposed sanctions or an embargo on B ("impose sanctions on", "A's sanctions on B", "B is under A sanctions"). Lifted, waived, threatened or called-for sanctions excluded. | 4 |
| `attacked` | A → B | Military/cyber attack (attack/strike/bomb/invade verbs, "A's attack on B", "Iranian attacks on B"); at least one side must be a place, so verbal "attacks" don't count. | 9 |
| `quoted_by` | A → news outlet B | B reports A's words: "A told Al Jazeera", "in an interview with B", "quoted by B". B must look like a news organisation. | 11 |
| `met_with` | symmetric | A met / held talks with B (meet verbs, "held talks with", "meeting between"). | 2 |
| `acquired` | A → ORG B | A bought/acquired B, or signed an agreement to ("Nvidia agrees to acquire Hugging Face"); talks, bids and plans excluded. | 2 |
| `invested_in` | A → ORG B | A invested in B, took a stake in B, funded a round B raised, or is described as B's investor ("OpenAI investor SoftBank", "SoftBank, an investor in OpenAI"); a money amount is required only for the "B raised $X from A" / "round led by A" forms. | 0 |
| `partnered_with` | symmetric, ORG–ORG | A and B partnered, teamed up, or signed a deal/partnership with each other. | 0 |
| `released` | ORG A → product B | A released/launched/unveiled a product named as the modifier of a product noun ("Google announces Gemini 4 Argon AI model"). | 1 |
| `discussed_topic` | A → TOPIC B | A discussed, negotiated, debated, warned about or announced something named by a topic phrase in the object (never a reason or setting). | 2 |
| `mentioned_with` | symmetric, **weak** | Same sentence, same segment, no semantic rule linked them; sentences with more than 8 entities (lists) add none. Stored with `quality_tier = cooccurrence`. | 1274 |

**Typed edges are the graph's primary layer; `mentioned_with` is a labelled weak tier.** In the final database
59 of 1,333 edges (4.4 %) are typed, almost all from news text (news: 68 typed evidence rows; Hacker News: 2;
Mastodon: 3, because comments rarely state "A did X to B"). Every edge in the API carries `relation_type`
and every citation its `quality_tier`, so a client can drop `cooccurrence` edges with one filter.

Every relation is extracted inside one sentence of one segment (never across comments). For event relations,
sentences that are negated ("did not meet"), hypothetical or planned ("will meet", "plans to acquire"),
questions, or reported rumours are suppressed. Each evidence row stores the rule id (e.g.
`affiliated_with.title`) and a quality tier, so precision can be measured per rule.

## Storage

SQLite ([db.py](src/media_intelligence/db.py), STRICT tables, foreign keys, numbered migrations):

| Table | Purpose |
| --- | --- |
| `nodes` | entity name, type, `first_seen`, `mention_count` |
| `edges` | source node, target node, relation type, `directed`, `weight`, `first_seen`, `last_seen` |
| `sources` | one row per observed version of a page: URL, type, title, body, author, published/scraped times, segments |
| `edge_sources` | edge ↔ source version, with the exact evidence sentence, segment id, offsets, rule id, `observed_at` |
| `node_mentions`, `node_aliases`, `crawl_runs` | per-source mentions, surface forms, run audit |

- **"According to what, and when?"** — every edge has at least one `edge_sources` row: the sentence, its
  location in the stored text, the page URL, publication time (if the page gave one) and when we observed it.
  `GET /edges/{id}/sources` returns them.
- `mention_count` and `weight` count **distinct source URLs**, so recrawls, edits and repeated sentences don't
  inflate them. `UNIQUE(source_url, content_hash)` makes identical reruns no-ops; an edited page becomes a new
  version, keeping history.
- Changing alias/topic/rule files needs a fresh database: `ingest` refuses to mix extractor versions.

## API

Real responses from the final database are saved in [examples/api/](examples/api/) (request, status, body):
`entities_central.json`, `network_iran_depth2.json`, `network_openai_depth1.json`,
`network_ambiguous_cloudflare_409.json`, `connections_new.json`, `edge_sources_example.json`, `health.json`.

- `GET /entity/{name}/network?depth=1|2` — breadth-first over incoming and outgoing edges; returns `nodes`
  (with distance) and `edges` (`source`/`target` ids, type, direction, weight, times, evidence link) ready for a
  graph renderer; deterministic truncation flag.
- `GET /connections/new?since=<ISO 8601 with timezone>` — new or significantly grown edges (below).
- `GET /entities/central` — ranked entities (below).
- Extra: `GET /edges/{id}/sources` (citations), `GET /health`, OpenAPI docs at `/docs`.

### "Grown significantly"

For an edge, each distinct source URL contributes once, at the time we first observed it supporting the edge.
Before `since`: B URLs; from `since` to now: D new URLs.

- **new**: B = 0 and D > 0.
- **grown**: B > 0, D ≥ 3 **and** D / B ≥ 0.5 (both configurable, returned in the response).

The absolute floor stops 1 → 2 counting as "+100 % growth"; the relative floor stops 100 → 103 counting as a
surge. Why 3 and 0.5: 84 % of edges in the final graph (1,123 of 1,333) have weight 1 and only 7 % (95) ever
reach 3, so three new independent URLs in one window is more support than most edges get in total; 0.5 means
the window added at least half the edge's prior support. In the run-3 window no edge with 3 or more new URLs
fell below the 50 % floor, so there the absolute floor did all the filtering.

**What it misses:** typed edges that double from a small base: `United States —attacked→ Iran` (#545) and
`Israel —attacked→ Iran` (#596) each went 2 → 4 URLs in the run-3 window and are not reported (140 existing
edges gained 1–2 URLs there); repetition inside one URL, so a story developing within a single long HN thread
never counts; already-large edges (needs +50 %). It also over-reports: "new" has no support floor, so 410 of
the 428 new edges in that window rest on a single URL, and all 22 "grown" edges are `mentioned_with`. It counts
our observations, not real-world publication, so a newly crawled old article looks "new"; it is not normalised
by window length or by how much we crawled; syndicated copies on different URLs count twice.

### Centrality

Normalised degree: distinct neighbours ÷ (N − 1), ignoring direction and counting a neighbour once even with
several relation types. Each result also reports relation-type diversity, weighted degree and mention count;
ties break on diversity, then weighted degree. **Measures:** who is connected to many different entities in
this corpus. **Misses:** intermediaries/bridges (no betweenness), credibility and real-world influence, and it
favours broad locations/topics and entities inflated by weak `mentioned_with` edges.

## Tests and verification

**Offline:** `uv run pytest -q` → 342 passed (config and frontier limits, each adapter on synthetic markup
modelled on the live pages, extraction rules with positive/suppressed/direction cases and the Musk case,
storage idempotence and versioning, growth boundaries and centrality ties, every API error code, offline
end-to-end ingest of all three adapters, the shipped config validating, no URL literals in `src/`). Synthetic
fixtures are labelled as such; none of the numbers below come from them.

**Live runs on the final database.** The evidence is committed under [examples/](examples/): the final database
`examples/final.sqlite3` (all ids in this README refer to it), the three run reports
`examples/runs/run-{1,2,3}-report.json`, the run-3 seed list `examples/run-3-sources.toml`, and saved API
responses in `examples/api/`. To query it without crawling:
`MI_DATABASE_PATH=examples/final.sqlite3 uv run uvicorn media_intelligence.api:app`. A fresh `ingest` re-crawls
live pages into `data/`, so its ids and counts will differ.

| Run | Config | Result |
| --- | --- | --- |
| 1 | shipped seeds, original limits (30 s timeout, 1 retry) | exit 3 (partial): 16 items stored; 4 pages failed during browser navigation (1 `TimeoutError`, 3 Crawl4AI `page.goto` errors): one seed per source plus one linked Al Jazeera article, recorded per source in the report |
| 2 | shipped `config/sources.toml` (timeout raised to 60 s, 2 retries) | exit 0: 5 new items (the 4 pages that failed in run 1, plus a changed version of one stored page), **15 unchanged**; the unchanged pages added no weight, and the 4 newly stored URLs raised the weight of 24 of the 852 existing edges |
| 3 | **changed seed list** (13 different seeds, same rules) | exit 0: 16 new, 2 unchanged (pages already seen) |

Totals: 37 stored versions of 35 distinct URLs (17 news articles stored as 19 versions, 8 discussion threads,
10 microblog threads). One Al Jazeera page is stored three times; its article text did not change, only its
rotating "Recommended Stories" list (see noise, below). 1,333 edges, 1,902 evidence rows,
`verify_integrity` clean. Evidence by source type: news 68 typed / 1,493 co-occurrence; discussion 2 / 321;
microblog 3 / 15. `/connections/new?since=2026-10-07T20:10:40Z` (start of run 3) returns 450 edges (428 new,
22 grown; examples of **grown**: `OpenAI — Anthropic` 2 → 7 distinct URLs, `Iran — military conflict` 8 → 13).

Earlier, during development (an earlier rule version on a 20-item calibration crawl), a manual review of all
51 semantic relations against their evidence sentences found 45 (88 %) correct; the rest were 3 merged-entity
errors (fixed later) and 3 definition stretches. The final `rules-4` graph has not been re-scored; edge #860
below is a known error.

**Fixed after the final runs (`rules-5`).** A review of the shipped database found three issues, now fixed in
code with regression tests built from the real failing text: (1) Al Jazeera's "Recommended Stories" widget
text is stripped from article bodies (`normalization.EMBEDDED_LIST`;
`test_embedded_recommendation_list_is_removed_from_article_body`); (2) `affiliated_with.appositive` requires
a PERSON subject, which removes edge #860 (`test_headline_comma_list_is_not_an_affiliation`); (3) reviewed
aliases fix Cloudflare and Docker as ORG, so each is one node (`test_reviewed_alias_fixes_company_type_across_items`).
`examples/final.sqlite3` and the numbers and ids in this README come from the `rules-4` live runs, so they still
show these issues; a new `ingest` uses `rules-5` (and needs a new database, because extractor versions are never
mixed).

## README: the hard questions

### 1. One real relationship, walked through

**Edge #860: `Nvidia —affiliated_with→ Supermicro`. It is wrong.**

- *Source:* Al Jazeera, <https://www.aljazeera.com/economy/2026/8/25/nvidia-supermicro-employees-charged-over-export-of-ai-servers-to-china>
  (`source_type = news`, published 2026-08-25T08:51:48Z, scraped 2026-10-07T20:08:57Z in crawl run 2; it
  was one of the AI-geopolitics seeds).
- *Evidence row* (`edge_sources.id = 1008`, returned by `GET /edges/860/sources`): segment `title`, offsets
  0–71, text **"Nvidia, Supermicro employees charged over export of AI servers to China"**, rule
  `affiliated_with.appositive`, tier `semantic_rule`.
- *What happened:* spaCy tagged `Nvidia` and `Supermicro` as ORG (Nvidia is also a reviewed alias entry;
  Supermicro is not) and parsed the headline as `Nvidia ←appos— employees`, with `Supermicro` as a compound
  modifier of `employees`. The appositive rule in `extraction.py` accepts any role noun from `ROLE_WORDS` and
  takes the organisation from the role's modifiers or an of/at/for phrase (as in "Esmaeil Baghaei, the Iranian
  Foreign Ministry spokesman", edge #204, which is correct). `employee` is in `ROLE_WORDS` and `Supermicro`
  modifies it, so the rule emitted Nvidia → Supermicro.
- *Why it is wrong:* in headline style the comma means "and" ("Nvidia [and] Supermicro employees"). Nvidia is
  not part of Supermicro; the article is about employees of both companies being charged.
- *Fix (done in `rules-5`):* `affiliated_with.appositive` now requires a PERSON subject (Nvidia is an ORG),
  checked by `test_headline_comma_list_is_not_an_affiliation` on this exact headline, while "Esmaeil Baghaei,
  the Iranian Foreign Ministry spokesman" still produces its edge. The evidence row made the error cheap to
  find: filtering `edge_sources` by `rule_id` and `segment_id = 'title'` lists every headline-derived edge.

For contrast, **edge #684 `Mark Rutte —affiliated_with→ NATO`** is correct and shows how weight works: rule
`affiliated_with.title` matched "NATO Secretary-General Mark Rutte strongly backed the US actions…" (Al
Jazeera, 2026-07-08 article, run 1) and later "NATO chief Mark Rutte called the US's latest attacks on Iran
“absolutely necessary”…" (a different article, crawled in batch 2, run 3). Two distinct URLs make
`weight = 2`; re-crawling either article again leaves it at 2.

### 2. How entity normalisation breaks

Real examples from the final database:

1. **First names don't reach full names across items.** In Hacker News comments, "if **Dario** and Sam can
   convince the US government to regulate and outlaw open source AI…" (comment 49427093 on
   <https://news.ycombinator.com/item?id=49411102>) and "Altman and **Dario** are not evil…" (comment 49906877
   on item 49905633) produce node #172 `Dario` (PERSON), separate from #197 `Dario Amodei`, which comes from
   Al Jazeera's "Anthropic CEO Dario Amodei". The same happens to `Sam` vs `Sam Altman`. *Why:* the
   surname/first-name rule only resolves a short name to a full name **inside the same item** (two-pass per
   item, deliberately, so "Musk" is not forced onto Elon Musk when the item names another Musk), and neither
   thread contains the full name. A global "Dario → Dario Amodei" alias would fix this case but would wrongly
   merge every other Dario; the right fix is a reviewed first-name alias scoped to a topic/source, or linking
   to a knowledge base.
2. **Type-aware identity splits one company.** Node keys are `TYPE:normalized name` (`models.node_key`), so a
   person and an organisation called the same thing never merge. The cost: spaCy tagged "Cloudflare" as ORG
   in one comment ("eventually put Cloudflare up") and as a place in another ("puts it in the Cloudflare of AI
   models category"), giving `Cloudflare (ORG)` #376 and `Cloudflare (LOCATION)` #410; `GET
   /entity/Cloudflare/network` returns **409** with both candidates. The per-item "one label per name" fix
   doesn't help across items; a reviewed alias entry (which fixes the type) does, and `rules-5` adds one for
   Cloudflare (and Docker, split the same way). The general problem remains for unreviewed names.
3. **Capital vs. country is kept apart on purpose, and that splits real relations.** `Tehran —attacked→
   United States` (#550, "Tehran's retaliatory attacks on US bases…") and `Iran —attacked→ United States`
   (#536) describe the same actor. `aliases.toml` deliberately does not alias a capital to its country (or a
   country to its government), because "Tehran" is also a city in plenty of sentences; the price is split
   weights and centrality.
4. **Products typed as people.** Model names such as "Qwen 3.8" (an HN comment) and "Claude Mythos Preview"
   (an Al Jazeera article) come out as PERSON; only reviewed product aliases are typed ORG.

What works, also from the real data: "Hawkins told Al Jazeera English's Tom McRae" resolved "Hawkins" to
`Tim Hawkins` (full name earlier in the same article), giving edge #792 `Tim Hawkins —quoted_by→ Al Jazeera`.
Reviewed handles such as `@openai` and `@realdonaldtrump` are stored in `node_aliases`, so the API resolves
them to the same node as the name. The PDF case ("Elon Musk",
"Musk", "@elonmusk" → one node) is a regression test (`tests/unit/test_extraction.py`).

### 3. Detecting and suppressing noise at scale

**Already in the implementation:**
- *Noise is labelled, not hidden.* Every evidence row has `quality_tier` (`semantic_rule` vs `cooccurrence`) and
  a `rule_id`. In the final database 95.6 % (1,274 of 1,333) of edges are `mentioned_with`; the run report splits typed
  vs. co-occurrence evidence per source type (`graph.evidence_by_source_type_and_tier`), which is how we saw
  that Hacker News/Mastodon comments yield almost only co-occurrence.
- *Locality.* Relations come from one sentence of one segment, so two comments in a thread are never joined;
  sentences listing more than 8 entities add no `mentioned_with` edges (stops lists from creating cliques).
- *Entity filters before edges.* Filters in `Extractor._candidates` (`_junk`, the `PHOTO_CREDIT` pattern,
  `_lowercase_span`, `_shouted_us`) and `Extractor._common_word` drop photo credits, list fragments, lowercase
  spans, capitalised common words and a shouted "US" pronoun; per-item one-label-per-name stops a name
  splitting into a PERSON and an ORG node; unmapped `@handles` never become nodes.
- *Assertion filters.* Negation, modality/plans, questions and reported rumours suppress event relations.
- *Counting.* `weight` counts distinct source URLs, so a recrawl, an edited version or a sentence repeated 20
  times in a thread adds nothing; `storage.verify_integrity` recomputes weights and mention counts from the
  evidence after every run (all zero violations in the final run).

**What I would add at scale (all built on columns that already exist):**
1. *Per-rule precision sampling.* Sample N evidence rows per `rule_id` each day, have a reviewer label them,
   and disable or demote any rule whose precision drops below a bar. This is how the rules here were tuned:
   the calibration review above measured 88 % precision and found the merged "Ministry of Foreign Affairs"
   node problem.
2. *Support thresholds by tier.* Hide `mentioned_with` edges in API responses unless `weight >= 2` (two
   distinct URLs) or they occur in two source types; keep them stored for recall.
3. *Hub detection.* Flag nodes whose co-occurrence degree is far above their typed degree (generic
   places/topics such as "United States" here) and down-weight them in centrality.
4. *Boilerplate fingerprints.* Sentences that recur on many pages of one site (newsletter boxes, "Read more")
   become per-source exclusions; the `exclude_selectors` config option is the first version of this, and it is
   not enough. On 6 of the 37 stored pages Al Jazeera's "Recommended Stories" list (other articles' headlines)
   still leaked into the body, giving 26 edges with evidence from those headlines (13 with no other evidence),
   e.g. `Ethiopia — Eritrea` (#533) cited to an article about an Iranian submarine. `rules-5` strips that
   widget's text (`normalization.EMBEDDED_LIST`); a per-site fingerprint of lines repeated across pages would
   catch the next such widget without code.
5. *Alias review queue.* `node_aliases` stores every surface form with its kind; surface forms shared by
   several nodes, or nodes where one name extends another (in the final data: "Docker Inc" (ORG) / "Docker"
   (PERSON), "US Department of the Treasury" / "Treasury"), go to a merge queue rather than being merged
   automatically.

### 4. Replacing SQLite with a graph database

**Easier with Neo4j:**
- *Traversal.* `analysis.network()` does breadth-first search by hand: one `_neighbour_ids()` SQL round trip per
  hop over a `json_each` id list, then a second query for the induced edges, then Python-side truncation. In
  Cypher that is one variable-length pattern (`MATCH p=(e {name:$n})-[*1..2]-(m)`), and depth 3+ or shortest
  paths would cost nothing extra to write.
- *Intermediaries.* Our centrality is degree by choice (one explainable SQL aggregate). Betweenness ("appears
  most frequently as intermediaries" in the brief) would mean loading the graph into Python and running
  Brandes' algorithm ourselves; Neo4j Graph Data Science provides betweenness and PageRank as built-in
  procedures.
- *Schema evolution for relations.* A new relation type is just a new relationship label; here it is a
  `relation_type` string we filter on.

**What we would lose or have to rebuild:**
- *Evidence as first-class rows.* Provenance lives in `edge_sources` (edge ↔ source version, sentence, segment,
  offsets, rule id, `observed_at`) with `UNIQUE(edge_id, source_id)`. In a property graph that becomes either
  evidence nodes (doubling traversal hops) or list properties on relationships (losing per-citation
  uniqueness and joins).
- *The growth query.* `/connections/new` is one SQL aggregate (`_GROWTH` in `analysis.py`): earliest
  observation per (edge, source URL), split at `since`, compared as exact fractions. It relies on GROUP BY over
  the evidence table; in Cypher it becomes collect/unwind over evidence nodes and is harder to verify.
- *Integrity guarantees.* STRICT tables, CHECK constraints (timestamp format, `weight >= 1`, symmetric edges
  stored in id order), foreign keys and one transaction per source item (`storage.store_item`). Neo4j has
  uniqueness/existence constraints but not this range of checks.
- *Zero-ops reproducibility.* The evaluator runs `uv sync` and gets a single database file; a graph database
  adds a server, credentials and a second query language.

At this corpus size (hundreds of nodes) SQLite is not the bottleneck; we would switch when traversal depth or
path/betweenness analysis becomes a product requirement.

### 5. Running continuously

What already supports repeated runs: `ingest` can be rerun on the same database. Unchanged pages are no-ops
(`UNIQUE(source_url, content_hash)`; `storage.store_item` only refreshes `last_scraped_at`/`last_seen`); an
edited page or a thread with new comments becomes a new source version. An edge this URL did not support
before gets its own `observed_at`, and that is what `/connections/new` counts; new comments repeating a pair
the thread already supported add nothing, because growth counts distinct URLs, so continuous mode would need a
per-segment or per-day count to see a story developing inside one thread. Each run is audited in `crawl_runs`
with a JSON report; SQLite runs in WAL mode so the API reads while one writer ingests.

What it would take:
1. **A scheduler and a single writer.** Today a run is one CLI process. Continuous mode needs a timer or worker
   loop with a lock so only one ingest writes (the code uses `BEGIN IMMEDIATE` per item but assumes one
   ingester).
2. **Discovery instead of fixed permalinks.** Seeds are mostly fixed thread/post URLs. Continuous crawling needs
   listing entry points re-polled on a schedule: the HN front page (already supported via `listing_paths` +
   `follow_links = "listings"`), RSS/sitemaps for Al Jazeera, and an authenticated API or tag timeline for
   Mastodon (logged-out tag pages expose almost no post links).
3. **A persistent frontier and revisit policy.** `crawler.Frontier` lives in memory for one run. We would store
   per-URL `next_visit` times (frequent for fresh threads, decaying for old articles) and conditional fetches,
   while keeping per-host politeness (HN `Crawl-delay: 30` makes it sequential and slow).
4. **Rule versioning.** `pipeline._refuse_mixed_versions` blocks mixing extractor versions in one database. A
   long-running graph needs a `rebuild` that re-extracts stored source versions with new rules, or versioned
   edges.
5. **Incremental analytics.** Centrality is recomputed on every request (`analysis.central`, a full pass over
   edges), which is fine here but would need caching or incremental degree updates at scale. Old evidence
   would need a retention/decay policy so "growth" reflects recent coverage.
6. **Monitoring.** Alert on `crawl_runs` status `partial`, on robots/bot-wall failures per source, and on sudden
   drops in semantic-edge share (a sign a site changed its markup and the adapter is now extracting junk).

## Limitations

- **Discussion and microblog text is mostly co-occurrence.** Comments rarely state "A did X to B" explicitly;
  of 323 discussion evidence rows only 2 are typed. Typed edges come overwhelmingly from news text.
- `invested_in` and `partnered_with` are implemented and tested but have no example in this corpus.
- Topic vocabulary is fixed (`topics.toml`), so new themes are missed; `discussed_topic` fires rarely.
- Growth counts our observations, not publication; all 22 "grown" edges in the run-3 window are
  `mentioned_with`, so they say two entities appear together more often, not why.
- Centrality is degree-based and dominated by broad locations (United States, Iran) and co-occurrence.
- Mastodon reply targets are not exposed in the rendered page, so replies are attached to the post; Mastodon
  seeds must be post URLs (logged-out profile/tag pages show almost no post links).
- Hacker News is slow to crawl on purpose (30 s between requests, per its robots.txt).
- Changing alias/topic/rule files requires a new database (no `rebuild` command).
