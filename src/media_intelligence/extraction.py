"""Local spaCy entities, reviewed aliases and topics, and explicit dependency rules for typed relations."""

import re
import tomllib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Literal

import spacy
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token
from spacy.util import filter_spans

from .models import ContentItem, EntityMention, Extraction, Relation, Segment, node_key, normalize_name

RULES_VERSION = "rules-5"
RULER_LABELS = {"PERSON": "PERSON", "ORG": "ORG", "LOCATION": "GPE"}
LABELS = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "LOCATION", "LOC": "LOCATION", "FAC": "LOCATION"}
# NORP is nationality/religion/political group ("Chinese", "Iranian"): not an organisation and not a place, so
# it would only add junk ORG nodes. NORP spans are kept only when a reviewed alias maps them ("Houthis").
ALIAS_ONLY_LABELS = {"NORP"}
# Labels a PERSON reading elsewhere in the item may override; a place label (GPE/LOC) is never turned into a person.
PERSON_RELABEL = {"ORG", "FAC", "NORP"}
TITLES = frozenset({
    "mr", "mrs", "ms", "miss", "dr", "sir", "dame", "president", "vice", "prime", "minister", "foreign", "defence",
    "defense", "finance", "interior", "secretary", "general", "gen", "state", "of", "the", "senator", "sen", "rep",
    "representative", "congressman", "congresswoman", "chancellor", "premier", "prince", "princess", "crown",
    "emir", "sheikh", "sultan", "ayatollah", "supreme", "leader", "ambassador", "envoy", "governor", "gov", "mayor",
    "judge", "justice", "chief", "commander", "lt", "col", "colonel", "lieutenant", "captain", "capt", "admiral",
    "adm", "field", "marshal", "spokesman", "spokeswoman", "spokesperson", "deputy", "former", "ex", "late",
    "chairman", "lord", "lady", "king", "queen", "pope"
})
NAME_SUFFIXES = {"jr", "sr"}
SURNAME_PARTICLES = {"de", "da", "di", "du", "van", "von", "der", "den", "ter", "le", "la", "bin", "ibn", "al", "el"}
ABBREVIATION_SKIPS = {"of", "and", "the", "for", "on", "in", "at", "to", "de", "du", "des", "la", "le"}
# Portfolio words the small model tags as ORG in "Trade Minister X", and Reddit/X jargon or platform names it
# tags as ORG/PERSON ("OP", "TIL", "X", "Twitter"). As a whole single-word entity, none is a geopolitical actor.
PORTFOLIOS = {"trade", "health", "energy", "economy", "education", "agriculture", "commerce", "labour", "labor",
              "culture", "transport", "environment", "industry", "information", "communications", "development",
              "security", "intelligence", "home"}
JARGON = {"op", "til", "ama", "nsfw", "eli5", "imo", "imho", "edit", "tldr", "lol", "lmao", "fwiw", "afaik", "iirc",
          "update", "x", "twitter", "reddit"}
# Technical acronyms the small model tags as ORG/GPE in AI threads ("LOCATION:ai", "ORG:llm", "ORG:cto"). As a whole
# single-word entity none of them names an organisation or a place; AI themes are covered by topics.toml instead.
TECH_ACRONYMS = {"ai", "agi", "asi", "llm", "llms", "api", "apis", "gpu", "gpus", "cpu", "cpus", "tpu", "tpus", "ram",
                 "ip", "cto", "ceo", "cfo", "coo", "ddos", "saas", "ml", "vc", "vcs", "ipo", "sdk", "ui", "ux", "os",
                 "pr", "faq", "url", "oss", "rag", "rlhf", "rl", "rce", "tos", "va", "ssd", "dry", "sota", "nda",
                 # "MoU" (memorandum of understanding) heads "Major MOU Violations"; HTML/HPC/EV/LNG/UTC are not actors.
                 "mou", "html", "css", "json", "hpc", "ev", "evs", "lng", "utc", "gmt", "gb", "tb", "csam", "vpn"}
ROLE_WORDS = frozenset({
    "president", "chief", "head", "leader", "secretary", "general", "minister", "spokesperson", "spokesman",
    "spokeswoman", "director", "chairman", "chairwoman", "chair", "ambassador", "envoy", "commander", "member",
    "official", "ceo", "founder", "deputy", "representative", "executive", "governor", "premier", "chancellor",
    "commissioner", "negotiator", "adviser", "advisor", "officer", "delegate", "lawmaker", "senator", "boss",
    "coordinator", "rapporteur", "administrator", "cofounder", "cto", "cfo", "coo", "scientist", "researcher",
    "engineer", "employee", "staffer", "correspondent", "editor", "reporter", "analyst"
})
ROLE_PREFIXES = {"co", "vice"}  # "co-founder", "vice-president" are split by the tokenizer
# Investor nouns: "SoftBank, an investor in OpenAI", "Microsoft is OpenAI's biggest investor" (invested_in, not a role).
INVESTOR_WORDS = {"investor", "backer", "shareholder"}
FORMER = {"former", "ex", "erstwhile"}
GENERIC = {"state", "states", "government", "administration", "ministry", "department", "council", "parliament",
           "team"}
# Every country has a "Foreign Ministry": a bare body name would merge Iran's, China's and Egypt's into one node. Such
# a name is qualified with the country written next to it ("Iran's Ministry of Foreign Affairs", "the Iranian
# Foreign Ministry" -> "Foreign Ministry (Iran)") and dropped when the sentence does not name one.
GOVERNMENT_BODY = re.compile(
    r"^(?:(?P<a>foreign affairs|foreign|defen[cs]e|interior|finance|energy|health|justice|information|oil|economy|"
    r"trade|commerce) ministry|ministry of (?:the )?(?P<b>foreign affairs|defen[cs]e|interior|finance|energy|health|"
    r"justice|information|oil|economy|trade|commerce))$")
PORTFOLIO_NAMES = {"foreign affairs": "Foreign", "defense": "Defence"}
POSSESSOR = re.compile(r"^(?P<owner>.+?)['’]s\s+(?P<body>.+)$")
# Photo credits at the end of captions ("[US Central Command/Handout via Reuters]") name no actor of the story.
PHOTO_CREDIT = re.compile(r"\[[^\[\]\n]*(?:/|\bvia\b|\bHandout\b|\bGetty\b|\bAP Photo\b)[^\[\]\n]*\]")
# Company-form suffixes written with or without a final period ("Shipoil Ltd" / "Shipoil Ltd.").
COMPANY_SUFFIX_DOT = re.compile(r"\b(Ltd|Inc|Co|Corp|Plc|Pte|Bhd|Llc|LLC)\.$")
NAME_BREAKERS = set("()[]{}/+|")  # a list or a fragment ("Lego (Denmark", "ML/AI"), never one name
MEET_VERBS = {"meet", "host"}
HOLD_VERBS = {"hold", "have", "conduct"}
MEET_NOUNS = {"meeting", "talk", "summit"}
CRITICIZE_VERBS = {"criticize", "criticise", "condemn", "denounce", "slam", "accuse", "blame", "rebuke", "censure",
                   "decry", "deplore", "lambast", "lambaste", "castigate"}
# Communication verbs whose object or about/on/over/regarding complement names a topic (discussed_topic).
DISCUSS_VERBS = {"discuss", "negotiate", "debate", "address", "talk", "warn", "comment", "speak", "testify",
                 "announce"}
DISCUSS_NOUNS = {"talk", "negotiation", "discussion", "debate", "warning"}
LEAD_VERBS = {"lead", "head", "chair"}  # person subjects only, like "run"
PERSON_LEAD_VERBS = {"run"}  # "Altman runs OpenAI"; an organisation "running" another is usually a product or service
FOUND_VERBS = {"found", "cofound", "co-found"}
WORK_VERBS = {"work"}  # "X works at/for Google"
TOPIC_PREPS = {"about", "on", "over", "regarding"}
INNER_TOPIC_PREPS = {"of", "about", "regarding"}  # inside a topic phrase "on"/"over" mark a target or a reason
SETTING_NOUNS = {"sideline", "sidelines", "margin", "margins", "eve", "wake", "backdrop", "occasion", "heels"}
# acquired: A bought (or signed an agreement to buy) organisation B.
ACQUIRE_VERBS = {"acquire", "buy", "purchase"}
ACQUIRE_NOUNS = {"acquisition", "purchase", "takeover", "buyout"}
# "Nvidia agrees to acquire X", "a deal to buy X": a signed acquisition agreement counts as an announced deal. Any
# other governing predicate (plan, seek, talks, bid, offer) still suppresses, and so does a refusal above "agree".
ANNOUNCED_DEAL_HEADS = frozenset({"agree", "sign", "deal", "agreement"})
# invested_in: A put money into B (investment, stake, funding round).
INVEST_VERBS = {"invest"}
INVEST_NOUNS = {"investment", "stake"}
STAKE_VERBS = {"take", "buy", "acquire", "hold", "purchase"}
RAISE_VERBS = {"raise", "secure"}
MONEY_WORDS = {"funding", "round", "capital", "money", "investment", "financing", "cash", "billion", "million", "bn",
               "m", "b"}
# partnered_with: A and B formed a partnership or signed/announced a deal with each other (symmetric, ORG-ORG).
PARTNER_VERBS = {"partner", "collaborate"}
DEAL_VERBS = {"sign", "strike", "reach", "ink", "seal", "announce", "form", "enter", "have", "agree", "unveil"}
DEAL_NOUNS = {"partnership", "deal", "agreement", "collaboration", "alliance", "pact", "contract", "tie-up"}
# released: organisation A released or launched product B. Products are typed ORG (aliases.toml), so B must be named
# as the modifier of a product noun ("announces Gemini 4 Argon AI model"): "Houthis released the Galaxy Leader" (a
# freed ship) or "launched Operation X" never qualify.
RELEASE_VERBS = {"release", "launch", "unveil", "announce", "ship", "debut", "introduce", "open-source", "roll"}
PRODUCT_NOUNS = {"model", "chatbot", "assistant", "app", "application", "chip", "processor", "gpu", "version",
                 "feature", "tool", "browser", "product", "device", "phone", "service", "platform", "agent", "llm",
                 "api", "sdk", "framework", "library", "update", "upgrade", "system", "weights"}
# sanctioned: A imposed sanctions (or an embargo) on B.
SANCTION_VERBS = {"sanction", "blacklist"}
SANCTION_NOUNS = {"sanction", "embargo"}
# Nouns a "sanctions" compound turns into sanctions: "sanctions regime". Not "sanctions waiver" or "sanctions relief".
SANCTION_HEADS = {"regime", "package", "programme", "program", "campaign", "measure", "designation"}
MEMBER_PREPS = ("including",)  # "sanctioned several individuals, including X and Y"
IMPOSE_VERBS = {"impose", "reimpose", "announce", "unleash", "slap", "levy", "place", "issue", "introduce", "adopt",
                "expand", "extend", "tighten"}
# attacked: A carried out a military or cyber attack on B.
ATTACK_VERBS = {"attack", "strike", "bomb", "invade", "shell", "hack"}
ATTACK_NOUNS = {"attack", "strike", "airstrike", "bombing", "assault", "offensive", "raid", "invasion", "war",
                "bombardment", "counterattack", "cyberattack"}
OF_TARGET_NOUNS = {"invasion", "bombing", "bombardment"}  # "the invasion of Ukraine", "the bombing of Gaza"
LAUNCH_VERBS = {"launch", "carry", "conduct", "stage", "mount", "unleash", "wage"}
# A target named through what it owns: "attacks on US bases", "Iran's attack on the US Navy warship".
MILITARY_ASSETS = {"base", "force", "troop", "asset", "vessel", "ship", "warship", "tanker", "embassy", "consulate",
                   "personnel", "position", "facility", "installation", "soldier", "outpost", "drone", "aircraft",
                   "jet", "submarine"}
# A person who represents a country at an organisation ("Russia's deputy UN envoy") is not affiliated with it.
REPRESENTATIVE_ROLES = {"envoy", "ambassador", "representative", "delegate", "emissary"}
# "Oman's Foreign Ministry released a statement condemning Iran's attacks": the statement's issuer is the actor.
STATEMENT_NOUNS = {"statement", "post", "message", "letter", "communique", "tweet", "video", "speech"}
EVENT_TARGET_PREPS = ("on", "against")
# Sanctions governed by these verbs are being removed or contested, not imposed: "lifted sanctions on Iran".
LIFT_VERBS = {"lift", "waive", "ease", "remove", "revoke", "suspend", "end", "relax", "drop", "scrap", "rescind",
              "loosen", "reverse"}
QUANTITY_HEADS = {"dozen", "dozens", "hundreds", "series", "wave", "set", "round", "slew", "raft", "package", "number",
                  "string", "barrage", "batch", "spate", "list"}  # "dozens of strikes", "a new set of sanctions"
RELATIVE_PRONOUNS = {"who", "which", "that"}
# quoted_by: speaker A's words are reported by news outlet B. An organisation counts as a news outlet when a word of
# its canonical name is one of these (there is no MEDIA entity type); "told the UN Security Council" is not a quote.
# "agency", "press" and "guardian" are left out: the IAEA, the National Security Agency, the National Press Club and
# Iran's Guardian Council are not news outlets. Outlets whose name has none of these words are listed whole.
NEWS_WORDS = {"news", "times", "post", "tv", "television", "radio", "broadcasting", "journal", "tribune", "herald",
              "gazette", "jazeera", "reuters", "bloomberg", "bbc", "cnn", "cnbc", "afp", "presse", "irna", "tasnim",
              "fars", "axios", "politico", "telegraph", "economist", "techcrunch", "wired", "verge", "technica",
              "media", "outlet"}
NEWS_NAMES = {"associated press", "the associated press", "ap", "guardian", "the guardian", "press tv", "nyt",
              "new york times"}
QUOTE_VERBS = {"tell", "speak", "say", "quote", "cite", "confirm"}
OUTLET_NOUNS = {"agency", "outlet", "network", "channel", "broadcaster", "newspaper", "daily"}
SPOKES_WORDS = {"spokesperson", "spokesman", "spokeswoman", "official", "representative"}
INTERVIEW_NOUNS = {"interview", "conversation", "podcast"}
SUPPRESSING_HEADS = {"plan", "expect", "set", "refuse", "deny", "hope", "want", "intend", "seek", "agree", "aim",
                     "propose", "offer", "threaten", "decline", "prepare", "schedule", "try", "attempt", "promise",
                     "vow", "pledge", "wish", "likely", "unlikely", "due", "reject", "fail", "consider", "urge",
                     "ask", "invite", "need", "avoid", "refrain", "hesitate", "ready", "willing", "reluctant",
                     "unwilling", "press", "push", "pressure", "encourage", "persuade", "demand", "postpone", "delay",
                     "cancel", "claim", "allege"}
# A finite clause attached to these nouns reports hearsay, not an event: "rumours that Nvidia bought Hugging Face".
HEARSAY_NOUNS = {"rumour", "rumor", "report", "claim", "allegation", "speculation", "suggestion"}
INFINITIVE_HEADS = {"have", "go"}  # only above an infinitive with "to": "has to meet", "is going to meet"
# Predicates whose subject event has not happened: "talks were postponed", "the meeting is expected to ...".
UNREALIZED_HEADS = {"postpone", "delay", "cancel", "plan", "schedule", "expect", "propose", "set"}
FOR_HEADS = {"call", "push", "press", "appeal", "ask", "hope", "wait", "prepare", "plan", "lobby", "demand"}
PLANNED_MODIFIERS = {"planned", "proposed", "possible", "potential", "upcoming", "expected", "scheduled", "future",
                     "cancelled", "canceled", "postponed", "attempted", "failed", "aborted", "abandoned", "rumoured",
                     "rumored", "blocked", "hypothetical", "alleged"}
PLANNED_LEMMAS = {"plan", "schedule", "propose", "expect", "postpone", "cancel", "delay"}
CONDITIONAL_MARKS = {"if", "unless", "whether"}
# "deterred Tehran from attacking", "refrained from criticising": the event under "from" did not happen.
FROM_HEADS = {"deter", "prevent", "stop", "refrain", "bar", "ban", "block", "keep", "prohibit", "dissuade", "discourage",
              "restrain", "abstain"}
# Phrasal refusals: "turned down an investment", "called off the deal", "ruled out talks", "backed out of".
PHRASAL_SUPPRESSORS = {("turn", "down"), ("call", "off"), ("rule", "out"), ("back", "out"), ("pull", "out")}
BLOCKING_DETERMINERS = {"no", "any"}  # "no meeting", "any US strike would ..."
DIRECTED = {"affiliated_with", "criticized", "discussed_topic", "acquired", "invested_in", "sanctioned", "attacked",
            "quoted_by", "released"}
# Relation definitions (README quotes this table). Direction is source -> target; symmetric relations store the pair
# in canonical key order. Each rule_id names the construction that produced the edge.
RELATION_DEFINITIONS = {
    "met_with": "Symmetric. A and B met or held talks (meet verbs, A hosted person B, held talks with, meeting "
                "between/with).",
    "criticized": "A -> B. A criticized, condemned, accused or blamed B, or condemned an action attributed to B "
                  "(the PDF's accused_of is folded in here: 'A accused B of X' is criticized, X is not stored).",
    "affiliated_with": "A -> ORG B. A holds a current role in or works for organisation B, a person leads or runs "
                       "B, or A founded B (titles, appositives, present-tense copulas, 'B's A', 'A of B', "
                       "lead/run/found/work-for verbs). Former roles are excluded.",
    "discussed_topic": "A -> TOPIC B. A discussed, negotiated, debated, warned about, commented on, spoke about or "
                       "announced something named by topic B; the topic must be in the object (or an about/on/over "
                       "complement), never a reason or a setting. Announced sanctions or strikes are typed "
                       "sanctioned/attacked instead.",
    "acquired": "A -> ORG B. A bought or acquired organisation B, or signed an agreement to (acquire/buy/purchase, "
                "A's acquisition of B, B was acquired by A).",
    "invested_in": "A -> ORG B. A invested money in B, took a stake in B, funded a round B raised, or is "
                   "described as B's investor.",
    "released": "ORG A -> ORG B (a product). A released, launched, unveiled, shipped or announced product B, named as "
                "the modifier of a product noun ('Google announces Gemini 4 Argon AI model'); products are typed ORG.",
    "partnered_with": "Symmetric, ORG-ORG. A and B partnered, teamed up, or signed/reached/announced a deal, "
                      "partnership or agreement with each other.",
    "sanctioned": "A -> B. A imposed sanctions or an embargo on B (sanction verb, impose sanctions on/against, "
                  "A's sanctions on B, B is under A sanctions).",
    "attacked": "ORG/LOCATION A -> ORG/LOCATION B, at least one of them a place. A carried out a military (or cyber) "
                "attack on B (attack/strike/bomb/invade verbs, launched strikes on, A's attack on B, Iranian attacks "
                "on B, A's invasion of B); 'attacked B over/for X' is criticism and excluded.",
    "quoted_by": "PERSON/ORG A -> news outlet B. B reports A's words: A told/spoke to B, A said in an interview with "
                 "B, A was quoted or cited by B.",
    "mentioned_with": "Symmetric, weak. A and B occur in the same sentence of the same segment and no semantic rule "
                      "linked them; a sentence with more than 8 distinct entities (a list) adds none.",
}
MAX_SENTENCE_ENTITIES = 8
MAX_EVIDENCE_CHARS = 500
MAX_NAME_CHARS = 80
HANDLE = re.compile(r"(?<![A-Za-z0-9_@.])@([A-Za-z0-9_]{1,30})")
URLISH = re.compile(r"://|^www\.|\.(?:com|org|net|gov|io)\b", re.IGNORECASE)


@Language.component("newline_sentences")
def newline_sentences(doc: Doc) -> Doc:
    """Normalized text uses a line break only between blocks (paragraphs, list items, lines of a post)."""
    for tok in doc[1:]:
        if "\n" in doc[tok.i - 1].text or "\n" in doc[tok.i - 1].whitespace_:
            tok.is_sent_start = True
    return doc


class AliasEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical: str = Field(min_length=1)
    type: Literal["PERSON", "ORG", "LOCATION"]
    aliases: list[str] = Field(default_factory=list)
    handles: list[str] = Field(default_factory=list)


class TopicEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    patterns: list[str] = Field(min_length=1)


def load_aliases(data: dict) -> tuple[list[AliasEntry], dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Entries plus name and handle lookup tables; one alias mapping to two entities is a configuration error."""
    entries = [AliasEntry.model_validate(e) for e in data.get("entities", [])]
    names: dict[str, tuple[str, str]] = {}
    handles: dict[str, tuple[str, str]] = {}
    for entry in entries:
        target = (" ".join(entry.canonical.split()), entry.type)
        for table, values in ((names, [entry.canonical, *entry.aliases]), (handles, entry.handles)):
            for value in values:
                key = normalize_name(value.removeprefix("@"))
                if not key or table.setdefault(key, target) != target:
                    raise ValueError(f"Alias {value!r} is empty or maps to more than one entity")
    return entries, names, handles


def reviewed_aliases(entries: list[AliasEntry]) -> dict[str, list[tuple[str, str]]]:
    """Node key -> (alias key, kind) for every configured alias and handle, so storage can record why names map."""
    table: dict[str, dict[str, str]] = {}
    for entry in entries:
        aliases = table.setdefault(node_key(entry.type, entry.canonical), {})
        for value in entry.aliases:
            kind = "abbreviation" if value.replace(".", "").isupper() else "alias"
            aliases.setdefault(normalize_name(value), kind)
        for value in entry.handles:
            # Handles resolve with or without the "@" that marks them in text.
            handle = normalize_name(value.removeprefix("@"))
            aliases.setdefault("@" + handle, "handle")
            aliases.setdefault(handle, "handle")
        aliases.pop(normalize_name(entry.canonical), None)
    return {key: sorted(aliases.items()) for key, aliases in table.items()}


def load_topics(data: dict) -> list[TopicEntry]:
    topics = [TopicEntry.model_validate(t) for t in data.get("topics", [])]
    seen: dict[str, str] = {}
    for topic in topics:
        for pattern in topic.patterns:
            if seen.setdefault(normalize_name(pattern), topic.name) != topic.name:
                raise ValueError(f"Topic pattern {pattern!r} belongs to more than one topic")
    return topics


@dataclass
class _Mention:
    span: Span
    start: int
    end: int
    label: str
    entity_type: str | None = None
    canonical: str | None = None
    resolution: str = "exact"
    reviewed: bool = False  # resolved through aliases.toml, which also fixes the type

    @property
    def key(self) -> str:
        return node_key(self.entity_type or "", self.canonical or "")

    @property
    def surface(self) -> str:
        return normalize_name(self.span.text)


class Extractor:
    def __init__(self, model_name: str, aliases_path: Path, topics_path: Path):
        alias_raw, topic_raw = Path(aliases_path).read_bytes(), Path(topics_path).read_bytes()
        entries, self.names, self.handles = load_aliases(tomllib.loads(alias_raw.decode()))
        topics = load_topics(tomllib.loads(topic_raw.decode()))
        self.nlp = spacy.load(model_name)
        if "parser" in self.nlp.pipe_names:
            self.nlp.add_pipe("newline_sentences", before="parser")
        # Reviewed multi-word names and abbreviations are pinned before statistical NER (case-sensitive, so the
        # pronoun "us" is never "US"). Single words are not: a pinned "Trump" would split "Melania Trump"; they are
        # added after NER instead, only where NER found nothing and no other proper noun adjoins (_alias_tokens).
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([{"label": RULER_LABELS[e.type], "pattern": value} for e in entries
                            for value in (e.canonical, *e.aliases) if " " in value.strip() or value.isupper()])
        self.single_words = {value.strip(): RULER_LABELS[e.type] for e in entries for value in (e.canonical, *e.aliases)
                             if " " not in value.strip() and not value.isupper()}
        self.reviewed = reviewed_aliases(entries)
        # An NER span that is exactly a topic phrase ("Cybercrimes", "AI Safety") is a theme, not an organisation.
        self.topic_phrases = {normalize_name(p) for topic in topics for p in topic.patterns}
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for topic in topics:
            self.matcher.add(topic.name, [self.nlp.make_doc(p) for p in topic.patterns])
        digest = sha256(alias_raw + b"\0" + topic_raw).hexdigest()[:12]
        self.version = f"{RULES_VERSION}+{model_name}-{self.nlp.meta.get('version', '?')}+cfg-{digest}"

    @classmethod
    def from_config(cls, model_name: str, config_path: Path, aliases_file: str, topics_file: str) -> "Extractor":
        return cls(model_name, config_path.parent / aliases_file, config_path.parent / topics_file)

    def extract(self, item: ContentItem) -> Extraction:
        segments = item.segments or [Segment(id="body", text=item.body)]
        docs = list(self.nlp.pipe(s.text for s in segments))
        common = _lowercase_words(docs)
        found = [[m for m in self._candidates(doc) if not self._common_word(m, common)] for doc in docs]
        warnings: list[str] = []
        surnames, given = self._resolve_names([m for ms in found for m in ms])
        for doc, ms in zip(docs, found):
            ms += [m for m in _surname_tokens(doc, ms, surnames) if not self._common_word(m, common)]
            ms.sort(key=lambda m: m.start)
        _resolve_surnames([m for ms in found for m in ms], surnames, given, warnings)
        self._item_abbreviations(docs, found)
        for doc, ms in zip(docs, found):
            index = {i: m for m in ms if m.entity_type for i in range(m.span.start, m.span.end)}
            _qualify_government_bodies(ms, index, self.names)
        mentions, relations = [], []
        for segment, doc, ms in zip(segments, docs, found):
            kept = [m for m in ms if m.entity_type]
            mentions += [EntityMention(name=segment.text[m.start:m.end], entity_type=m.entity_type,
                                       canonical_name=m.canonical, key=m.key, segment_id=segment.id,
                                       start=m.start, end=m.end, resolution=m.resolution) for m in kept]
            tokens = {i: m for m in kept for i in range(m.span.start, m.span.end)}
            for sent in doc.sents:
                relations += _relations(segment, sent, tokens, warnings)
        linked = {frozenset((r.source_key, r.target_key)) for r in relations if r.quality_tier == "semantic_rule"}
        unique: dict[tuple[str, str, str], Relation] = {}
        for r in relations:
            if r.quality_tier == "cooccurrence" and frozenset((r.source_key, r.target_key)) in linked:
                continue
            unique.setdefault((r.source_key, r.target_key, r.relation_type), r)
        return Extraction(mentions=mentions, relations=list(unique.values()), warnings=list(dict.fromkeys(warnings)))

    def _candidates(self, doc: Doc) -> list[_Mention]:
        result, handle_spans = [], []
        for match in HANDLE.finditer(doc.text):
            handle_spans.append((match.start(), match.end()))
            target = self.handles.get(normalize_name(match.group(1)))
            span = doc.char_span(match.start(), match.end(), alignment_mode="expand")
            if target and span is not None:
                result.append(_Mention(span, match.start(), match.end(), "HANDLE", target[1], target[0], "handle"))
        # Handles and photo credits are skipped: neither is a mention of an actor in the text.
        excluded = handle_spans + [(m.start(), m.end()) for m in PHOTO_CREDIT.finditer(doc.text)]
        for ent in doc.ents:
            if ent.label_ not in LABELS and ent.label_ not in ALIAS_ONLY_LABELS:
                continue
            span = _trim(ent)
            if (span is None or _junk(span, ent.label_) or _shouted_us(span) or _tagged(span[0])
                    or normalize_name(span.text) in self.topic_phrases or self._lowercase_span(span)
                    or any(s < span.end_char and span.start_char < e for s, e in excluded)):
                continue
            result.append(_Mention(span, span.start_char, span.end_char, ent.label_))
        occupied = {i for m in result for i in range(m.span.start, m.span.end)}
        result += self._alias_tokens(doc, occupied, excluded)
        occupied = {i for m in result for i in range(m.span.start, m.span.end)}
        topics = [s for s in self.matcher(doc, as_spans=True) if s.root.pos_ not in ("VERB", "AUX")]
        for span in filter_spans(topics):
            if occupied.isdisjoint(range(span.start, span.end)):
                exact = normalize_name(span.text) == normalize_name(span.label_)
                result.append(_Mention(span, span.start_char, span.end_char, "TOPIC", "TOPIC", span.label_,
                                       "exact" if exact else "alias"))
        return sorted(result, key=lambda m: m.start)

    def _lowercase_span(self, span: Span) -> bool:
        """An all-lowercase span ("n’t", "punishment(monetary", "max") is not a proper name unless reviewed."""
        text = span.text
        return text == text.lower() and text != text.upper() and normalize_name(text) not in self.names

    def _common_word(self, m: _Mention, common: set[str]) -> bool:
        """A single capitalised word the same item also writes in lower case is a common noun read as a name
        ("Strikes", "Law", "Max" next to "strikes", "law", "max"), unless aliases.toml reviews it. All-caps
        acronyms ("OPT") are not affected."""
        text = m.span.text
        return (m.label not in ("HANDLE", "TOPIC") and len(m.span) == 1 and text[:1].isupper() and text[1:].islower()
                and text.lower() in common and m.surface not in self.names)

    def _alias_tokens(self, doc: Doc, occupied: set[int], handle_spans: list[tuple[int, int]]) -> list[_Mention]:
        """Reviewed single-word names NER missed ("Trump met Putin"): exact case, never inside a longer name."""
        return [_Mention(doc[t.i:t.i + 1], t.idx, t.idx + len(t), self.single_words[t.text]) for t in doc
                if t.text in self.single_words and t.i not in occupied and not _named_neighbour(t)
                and not _tagged(t) and not any(s <= t.idx < e for s, e in handle_spans)]

    def _resolve_names(self, mentions: list[_Mention]) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        """First pass: one label per name, exact/alias resolution, then index full person names by surname and by
        first name."""
        _harmonize_labels(mentions)
        surnames: dict[str, dict[str, str]] = {}
        given: dict[str, dict[str, str]] = {}
        for m in mentions:
            if m.label not in ("HANDLE", "TOPIC"):
                if target := self.names.get(m.surface):
                    m.canonical, m.entity_type = target
                    m.resolution = "exact" if m.surface == normalize_name(target[0]) else "alias"
                    m.reviewed = True
                elif m.label in LABELS:
                    m.canonical, m.entity_type = " ".join(m.span.text.split()), LABELS[m.label]
                    if m.entity_type == "ORG":  # "Shipoil Ltd." and "Shipoil Ltd" are one company
                        m.canonical = COMPANY_SUFFIX_DOT.sub(r"\1", m.canonical)
            # Only full names written in the item (or a mapped handle) can anchor a bare surname or first name.
            if m.entity_type == "PERSON" and (" " in m.surface or m.label == "HANDLE") and " " in m.canonical:
                full = normalize_name(m.canonical)
                parts, written = _name_parts(full), _name_parts(m.surface)
                tails = {parts[-1], written[-1]}
                if len(parts) > 2 and parts[-2] in SURNAME_PARTICLES:  # "Alfred-Maurice de Zayas" -> "de Zayas"
                    tails.add(" ".join(parts[-2:]))
                for last in tails - {m.surface}:
                    surnames.setdefault(last, {})[full] = m.canonical
                # Family-name-first names ("Wang Yi", "Kim Jong Un") are often shortened to their first word, but
                # an all-caps first word is a heading or an acronym ("MOU Violations"), not a given name.
                if not m.span.text.split()[0].isupper():
                    given.setdefault(parts[0], {})[full] = m.canonical
        return surnames, given

    def _item_abbreviations(self, docs: list[Doc], found: list[list[_Mention]]) -> None:
        """"Ministry of Intelligence and Security (MOIS)": the item defines MOIS, so every MOIS in the item is that
        name. Only an all-caps abbreviation whose letters follow the initials counts; reviewed aliases win."""
        defined: dict[str, _Mention] = {}
        for doc, ms in zip(docs, found):
            for m in ms:
                end = m.span.end
                if (m.entity_type in ("ORG", "LOCATION") and len(m.span) > 1 and end + 2 < len(doc)
                        and doc[end].text == "(" and doc[end + 2].text == ")"):
                    abbr = doc[end + 1].text
                    if (abbr.isalpha() and abbr.isupper() and 2 <= len(abbr) <= 8 and normalize_name(abbr) not in
                            self.names and _initials_match(abbr, m.canonical or "")):
                        defined.setdefault(abbr, m)
        if not defined:
            return
        for doc, ms in zip(docs, found):
            occupied = {i for m in ms for i in range(m.span.start, m.span.end)}
            ms += [_Mention(doc[t.i:t.i + 1], t.idx, t.idx + len(t), "ORG") for t in doc
                   if t.text in defined and t.i not in occupied]
            for m in ms:
                if m.span.text in defined and not m.reviewed and m.label not in ("HANDLE", "TOPIC"):
                    full = defined[m.span.text]
                    m.canonical, m.entity_type, m.resolution = full.canonical, full.entity_type, "abbreviation"
            ms.sort(key=lambda m: m.start)


def _lowercase_words(docs: list[Doc]) -> set[str]:
    return {t.text for doc in docs for t in doc if t.is_alpha and t.text == t.text.lower() and len(t) > 1}


def _harmonize_labels(mentions: list[_Mention]) -> None:
    """The small model labels the same name inconsistently within one item; give each written name one label.

    A PERSON reading wins over an organisation reading. A one-word name that is also read as a place is a person
    only when it is the first or last word of a full person name written in the item ("Altman" next to "Sam
    Altman"); otherwise the place wins ("At the Muscat meeting" next to "in Muscat"). Between an organisation and a
    place reading the majority wins, the organisation on a tie. Reviewed aliases fix the type later regardless.
    """
    places = {"GPE", "LOC", "FAC"}
    labels: dict[str, Counter] = {}
    for m in mentions:
        if m.label not in ("HANDLE", "TOPIC"):
            labels.setdefault(m.surface, Counter())[m.label] += 1
    name_parts = {part for m in mentions if m.label == "PERSON" and " " in m.surface
                  for part in (m.surface.split()[0], m.surface.split()[-1])}
    winners: dict[str, str] = {}
    for surface, seen in labels.items():
        place = sum(seen[label] for label in places)
        if len(seen) < 2:
            continue
        if seen["PERSON"] and place and " " not in surface:
            winners[surface] = "PERSON" if surface in name_parts else "GPE"
        elif seen["PERSON"]:
            winners[surface] = "PERSON"  # over ORG/FAC/NORP; a multi-word place reading is left as it is
        elif seen["ORG"] and place:
            winners[surface] = "ORG" if seen["ORG"] >= place else "GPE"
    for m in mentions:
        winner = winners.get(m.surface)
        if winner is None or m.label in ("HANDLE", "TOPIC"):
            continue
        if winner == "PERSON" and (m.label in PERSON_RELABEL or " " not in m.surface):
            m.label = "PERSON"
        elif winner == "GPE" and m.label not in places:
            m.label = "GPE"
        elif winner == "ORG":
            m.label = "ORG"


def _initials_match(abbr: str, name: str) -> bool:
    words = re.findall(r"[^\W\d_][^\W_]*", name)
    every = "".join(w[0] for w in words).lower()
    content = "".join(w[0] for w in words if w.lower() not in ABBREVIATION_SKIPS).lower()
    abbr = abbr.lower()
    return bool(every) and abbr[0] == every[0] and _subsequence(content, abbr) and _subsequence(abbr, every)


def _subsequence(short: str, long: str) -> bool:
    letters = iter(long)
    return all(c in letters for c in short)


def _qualify_government_bodies(mentions: list[_Mention], index: dict[int, _Mention], names) -> None:
    """Bare government bodies ("Foreign Ministry") get the country named next to them, or are dropped."""
    for m in mentions:
        if m.entity_type != "ORG" or m.reviewed:
            continue
        text, owner = normalize_name(m.canonical or ""), None
        if match := POSSESSOR.match(text):  # the model kept the owner inside the span: "Qatar's Foreign Ministry"
            text = match["body"]
            words = len(match["owner"].split())
            written = re.sub(r"['’]s?$", "", " ".join(m.span.text.split()[:words]))
            reviewed = names.get(match["owner"])
            owner = reviewed[0] if reviewed and reviewed[1] == "LOCATION" else written
        body = GOVERNMENT_BODY.match(text)
        if body is None:
            continue
        owner = owner or _owner_country(m.span, index)
        if owner is None:
            m.entity_type = None  # whose ministry is unknown, so it cannot be one shared node
            continue
        portfolio = body["a"] or body["b"]
        m.canonical = f"{PORTFOLIO_NAMES.get(portfolio, portfolio.title())} Ministry ({owner})"


def _owner_country(span: Span, index: dict[int, _Mention]) -> str | None:
    """The place written just before a body: "Iran's Ministry ...", "the Iranian Foreign Ministry", "Iran Foreign
    Ministry"."""
    doc, i = span.doc, span.start - 1
    if i >= 0 and doc[i].tag_ == "POS":
        i -= 1
    owner = index.get(i) if i >= 0 else None
    if owner is not None and owner.entity_type == "LOCATION":
        return owner.canonical
    return None


def _name_parts(name: str) -> list[str]:
    parts = name.split()
    return [p for p in parts if p.rstrip(".") not in NAME_SUFFIXES] or parts


def _tagged(tok: Token) -> bool:
    """A hashtag or handle ("#Iran", "@Trump"): a label chosen by the author, not a mention in the text."""
    return tok.idx > 0 and tok.doc.text[tok.idx - 1] in "#@"


def _named_neighbour(tok: Token) -> bool:
    """An adjacent proper noun other than a title means tok is part of a longer, different name."""
    doc = tok.doc
    return any(0 <= j < len(doc) and doc[j].pos_ == "PROPN" and doc[j].lower_.rstrip(".") not in TITLES
               for j in (tok.i - 1, tok.i + 1))


def _surname_tokens(doc: Doc, mentions: list[_Mention], surnames: dict[str, dict[str, str]]) -> list[_Mention]:
    """Bare surnames NER missed; a neighbouring proper noun means a different, unrecognized full name."""
    occupied = {i for m in mentions for i in range(m.span.start, m.span.end)}
    return [_Mention(doc[t.i:t.i + 1], t.idx, t.idx + len(t), "PERSON") for t in doc
            if t.pos_ == "PROPN" and t.i not in occupied and normalize_name(t.text) in surnames
            and not _named_neighbour(t)]


def _resolve_surnames(mentions: list[_Mention], surnames: dict[str, dict[str, str]],
                      given: dict[str, dict[str, str]], warnings: list[str]) -> None:
    """Second pass: a bare name (or a particle surname such as "de Zayas") resolves only to the single full name in
    the item that ends (or starts) with it.

    All-caps abbreviations and reviewed non-person names never take part: "UN" is not the "Un" of Kim Jong Un.
    """
    for m in mentions:
        if (m.label not in ("PERSON", "ORG") or (" " in m.surface and m.surface not in surnames)
                or m.span.text.isupper() or (m.reviewed and m.entity_type != "PERSON")):
            continue
        options = {**given.get(m.surface, {}), **surnames.get(m.surface, {})}
        if not options:
            continue
        if m.resolution == "alias":  # a reviewed single-word alias that disagrees with the item is a conflict
            options[normalize_name(m.canonical)] = m.canonical
        if len(options) > 1:
            m.entity_type = None
            names = ", ".join(sorted(options.values()))
            warnings.append(f"Ambiguous surname {m.span.text!r} left unresolved: {names}")
        elif m.resolution != "alias":
            m.canonical, m.entity_type, m.resolution = next(iter(options.values())), "PERSON", "surname"


def _trim(ent: Span) -> Span | None:
    doc, start, end = ent.doc, ent.start, ent.end
    while start < end and (doc[start].lower_ in ("the", "a") or doc[start].is_punct or doc[start].is_space):
        start += 1
    if end - start > 2 and doc[start].lower_ in FORMER and _joining_hyphen(doc[start + 1]):
        start += 2  # "Ex-Google" names Google; _former() still sees the prefix
    while end > start + 1 and doc[end - 1].lower_ in ("and", "or"):  # "Vienna Shipping Co and"
        end -= 1
    if ent.label_ == "PERSON":
        while start < end and doc[start].lower_.rstrip(".") in TITLES:
            start += 1
    while end > start and (doc[end - 1].is_punct or doc[end - 1].is_space or doc[end - 1].tag_ == "POS"):
        end -= 1
    # A pinned "Donald Trump" must not swallow "Donald Trump Jr.", a different person.
    if ent.label_ == "PERSON" and end > start and end < len(doc) and doc[end].lower_.rstrip(".") in NAME_SUFFIXES:
        end += 1
    return doc[start:end] if end > start else None


def _former_prefix(span: Span) -> bool:
    """"former NATO", "ex-Google" (the hyphenated prefix is trimmed off the entity span)."""
    doc, i = span.doc, span.start - 1
    if i >= 0 and _joining_hyphen(doc[i]):
        i -= 1
    return i >= 0 and doc[i].lower_ in FORMER


def _junk(span: Span, label: str = "") -> bool:
    text = span.text
    if len(span) == 1 and normalize_name(text).rstrip(".") in TITLES | ROLE_WORDS | PORTFOLIOS | JARGON | TECH_ACRONYMS:
        return True
    if label in ("PERSON", "ORG") and "," in text:  # a list the model read as one name: "Julien, Thomas"
        return True
    if any(c in NAME_BREAKERS for c in text):  # "ML/AI", "Lego (Denmark", "Murtaja Lateef/AFP"
        return True
    if label == "PERSON" and normalize_name(span[0].text) in TECH_ACRONYMS - ROLE_WORDS:  # "MOU Violations"
        return True
    return (len(text) < 2 or len(text) > MAX_NAME_CHARS or not any(c.isalpha() for c in text) or "\n" in text
            or normalize_name(text) in GENERIC or "@" in text or "#" in text or text.lower().startswith(("u/", "r/"))
            or bool(URLISH.search(text)))


def _shouted_us(span: Span) -> bool:
    """In an all-caps sentence ("LET US BE CLEAR"), a bare "US" is usually the pronoun, unless it is "THE US"."""
    if span.text != "US":
        return False
    words = [t for t in span.sent if t.is_alpha and len(t) > 1]
    shouting = sum(t.is_upper for t in words) >= 0.6 * len(words)
    return shouting and (span.start == span.sent.start or span.doc[span.start - 1].lower_ != "the")


def _children(tok: Token, *deps: str) -> list[Token]:
    return [c for c in tok.children if c.dep_ in deps]


def _pobjs(tok: Token, *preps: str) -> list[Token]:
    return [p for c in tok.children if c.dep_ == "prep" and c.lower_ in preps for p in _children(c, "pobj")]


def _agents(tok: Token) -> list[Token]:
    return [p for c in _children(tok, "agent") for p in _children(c, "pobj")]


def _infinitive(tok: Token) -> bool:
    return any(c.dep_ == "aux" and c.tag_ == "TO" for c in tok.children)


def _relative(tokens: list[Token], verb: Token) -> list[Token]:
    """A relative pronoun subject stands for the noun the clause modifies: "Altman, who runs OpenAI" -> Altman."""
    if verb.dep_ != "relcl":
        return tokens
    return [verb.head if t.lower_ in RELATIVE_PRONOUNS and t.tag_ in ("WP", "WDT") else t for t in tokens]


def _subjects(verb: Token) -> list[Token]:
    subjects = _relative(_children(verb, "nsubj"), verb)
    if subjects or verb.head.i == verb.i:
        return subjects
    if verb.dep_ == "xcomp":
        # Object control: in "Biden pressed Netanyahu to meet Abbas" the one meeting is Netanyahu.
        controllers = [o for o in _children(verb.head, "dobj") if o.pos_ in ("PROPN", "PRON")]
        return controllers or _subjects(verb.head)
    if verb.dep_ == "conj" or (verb.dep_ == "advcl" and _infinitive(verb)):  # "X met Y to discuss Z"
        return _subjects(verb.head)
    if verb.dep_ == "acl" and verb.tag_ == "VBD":  # a finite verb read as a modifier: "Apple partnered with ..."
        return [verb.head]
    if verb.dep_ == "advcl" and verb.tag_ == "VBG" and not _children(verb, "mark"):  # "X spoke, accusing Y of ..."
        return _subjects(verb.head)
    if (verb.dep_ == "acl" and verb.tag_ == "VBG" and verb.head.lemma_.lower() in STATEMENT_NOUNS
            and verb.head.dep_ == "dobj"):  # "X released a statement condemning Y"
        return _subjects(verb.head.head)
    return subjects


def _passive_subjects(verb: Token) -> list[Token]:
    return _relative(_children(verb, "nsubjpass"), verb)


def _ents(tokens: Iterable[Token], index: dict[int, _Mention]) -> list[_Mention]:
    """Entities denoted by head tokens, following conjunctions and appositive entities (lists like "The US, the UK,
    France and Japan" are parsed as an appositive chain); an appositive naming the same entity adds nothing."""
    found = []
    for tok in tokens:
        if tok.i in index:
            found.append(index[tok.i])
        found += _ents([c for c in _children(tok, "appos") if c.i in index], index)
        found += _ents(_children(tok, "conj"), index)
    return found


def _blocked(tok: Token) -> bool:
    """Negation, modality, a conditional, a "no"/"any" determiner, or a planned/cancelled modifier on this token."""
    return any(c.dep_ == "neg" or (c.dep_ == "aux" and c.tag_ == "MD")
               or (c.dep_ == "mark" and c.lower_ in CONDITIONAL_MARKS)
               or (c.dep_ == "det" and c.lower_ in BLOCKING_DETERMINERS)
               or (c.dep_ in ("amod", "acl") and (c.lower_ in PLANNED_MODIFIERS or c.lemma_.lower() in PLANNED_LEMMAS))
               for c in tok.children)


def _phrasal_refusal(tok: Token) -> bool:
    return any((tok.lemma_.lower(), c.lower_) in PHRASAL_SUPPRESSORS for c in tok.children if c.dep_ == "prt")


def _suppressed(tok: Token, allowed: frozenset[str] = frozenset()) -> bool:
    """Negated, modal, conditional, planned, refused, denied, or merely demanded events are not asserted.

    Climbs from the trigger through complements ("refused to meet"), its governing predicate when an event noun is
    the subject ("the meeting never happened", "talks were postponed", "any strike would ...") or a prepositional
    object ("called for talks", "deterred X from attacking"), stopping at a conjunct with its own subject. An
    infinitive attached to a noun ("talks to acquire", "a plan to meet") is unrealized. ``allowed`` names governing
    lemmas a rule accepts as asserted (an acquisition "agreed to" is an announced deal); the climb continues past
    them, so "refused to agree to acquire" is still suppressed.
    """
    for _ in range(8):
        if _blocked(tok) or _phrasal_refusal(tok):
            return True
        head, dep = tok.head, tok.dep_
        if head.i == tok.i:
            return _infinitive(tok)  # a root infinitive is headline/plan style: "Biden to meet Xi"
        lemma = head.lemma_.lower()
        if dep in ("nsubj", "nsubjpass"):
            if tok.pos_ != "NOUN":
                return False
            if lemma in UNREALIZED_HEADS:
                return True
        elif dep == "prep":
            if (tok.lower_ == "for" and lemma in FOR_HEADS) or (tok.lower_ == "from" and lemma in FROM_HEADS):
                return True
        elif dep == "acl":
            if (_infinitive(tok) and lemma not in allowed) or (head.pos_ == "NOUN" and lemma in HEARSAY_NOUNS):
                return True
            if not _infinitive(tok):
                return False
        elif dep == "xcomp" or (dep == "advcl" and _infinitive(tok)):
            if lemma in allowed:
                pass
            elif lemma in SUPPRESSING_HEADS or (lemma in INFINITIVE_HEADS and _infinitive(tok)):
                return True
        elif dep in ("ccomp", "dobj", "pcomp", "conj"):
            if dep == "conj" and _children(tok, "nsubj", "nsubjpass"):
                return False
            if lemma in SUPPRESSING_HEADS and lemma not in allowed:
                return True
            if dep == "ccomp" and head.pos_ == "NOUN" and lemma in HEARSAY_NOUNS:
                return True
        elif dep != "pobj":
            return False
        tok = head
    return False


CLAUSE_DEPS = {"relcl", "acl", "advcl", "ccomp", "xcomp", "parataxis", "punct"}


def _phrase(tok: Token) -> Iterator[Token]:
    """The noun phrase headed by tok without clauses or adjuncts: in "sanctions on Iran over its nuclear programme"
    neither the target nor the reason is what was announced. Compounds, modifiers, conjuncts and of/about/regarding
    complements stay ("the issue of sanctions", "a warning about human extinction")."""
    yield tok
    for child in tok.children:
        if child.dep_ in CLAUSE_DEPS or (child.dep_ == "prep" and child.lower_ not in INNER_TOPIC_PREPS):
            continue
        yield from _phrase(child)


def _prep_phrase(prep: Token) -> Iterator[Token]:
    """The object of a topic preposition, unless it names a setting: "spoke on the sidelines of the nuclear talks"."""
    for obj in _children(prep, "pobj"):
        if obj.lemma_.lower() not in SETTING_NOUNS and obj.lower_ not in SETTING_NOUNS:
            yield from _phrase(obj)


def _region_topics(tokens: Iterable[Token], index: dict[int, _Mention], shallow: Iterable[Token] = ()) -> list:
    region = {t.i for tok in tokens for t in (_prep_phrase(tok) if tok.dep_ == "prep" else _phrase(tok))}
    region |= {t.i for t in shallow}
    return [m for i, m in index.items() if i in region and m.entity_type == "TOPIC"]


def _role_orgs(role: Token, index: dict[int, _Mention], preps: tuple[str, ...] = ("of", "at", "for")) -> list[_Mention]:
    if any(c.lower_ in FORMER for c in role.children):
        return []
    modifiers = [t for left in role.lefts for t in left.subtree]
    found = [index[t.i] for t in modifiers if t.i in index] + _ents(_pobjs(role, *preps), index)
    if role.lemma_.lower() in REPRESENTATIVE_ROLES and any(m.entity_type == "LOCATION" for m in found):
        return []  # "Russia's deputy UN envoy" represents Russia at the UN
    return [m for m in found if m.entity_type == "ORG"]


def _actor_modifier(tok: Token, index: dict[int, _Mention]) -> bool:
    """An organisation, or an abbreviated place ("US", "UK"), modifying an event noun names who acted."""
    mention = index.get(tok.i)
    return mention is not None and (mention.entity_type == "ORG" or (
        mention.entity_type == "LOCATION" and mention.span.text.replace(".", "").isupper()))


def _place_modifier(tok: Token, index: dict[int, _Mention]) -> bool:
    """A title-case place modifying an event noun names its target: "Iran sanctions", "the Gaza attack"."""
    mention = index.get(tok.i)
    return mention is not None and mention.entity_type == "LOCATION" and not _actor_modifier(tok, index)


def _event_actors(noun: Token, index: dict[int, _Mention]) -> list[Token]:
    """Who carried out an event noun: "Iran's attacks", "the attack by Israel", "the Hamas attack", "US sanctions",
    "Iranian attacks" (a nationality adjective resolved to its country), and "Israel strikes on Gaza" (a place
    modifier is the actor only when the noun names its target explicitly; otherwise "Iran sanctions" is a target)."""
    explicit = bool(_pobjs(noun, *EVENT_TARGET_PREPS))
    return (_children(noun, "poss") + _pobjs(noun, "by")
            + [c for c in _children(noun, "compound") if _actor_modifier(c, index)
               or (explicit and _place_modifier(c, index))]
            + [c for c in _children(noun, "amod") if c.i in index and index[c.i].entity_type == "LOCATION"])


def _target_places(noun: Token, index: dict[int, _Mention]) -> list[Token]:
    """A title-case place modifier is the target when no on/against target is written: "issued new Iran sanctions"."""
    if _pobjs(noun, *EVENT_TARGET_PREPS):
        return []
    return [c for c in _children(noun, "compound") if _place_modifier(c, index)]


def _event_objects(verb: Token) -> list[Token]:
    """Direct objects, looking through quantity heads: "launched dozens of strikes", "imposed a new set of sanctions"."""
    objects = _children(verb, "dobj")
    return objects + [p for o in objects if o.lemma_.lower() in QUANTITY_HEADS or o.pos_ == "NUM"
                      for p in _pobjs(o, "of")]


def _has_particle(tok: Token, particle: str) -> bool:
    return any(c.dep_ == "prt" and c.lower_ == particle for c in tok.children)


def _sanction_noun(tok: Token) -> bool:
    """"sanctions", "an embargo", or a noun they modify: "sanctions regime", "sanctions package"."""
    if tok.pos_ not in ("NOUN", "PROPN"):
        return False
    return tok.lemma_.lower() in SANCTION_NOUNS or tok.lower_ == "sanctions" or (
        tok.lemma_.lower() in SANCTION_HEADS and any(
            c.dep_ == "compound" and (c.lemma_.lower() in SANCTION_NOUNS or c.lower_ == "sanctions")
            for c in tok.children))


def _members(tokens: list[Token]) -> list[Token]:
    """Targets plus the named members of a group target: "60 entities, including RPT Technology Ltd"."""
    return tokens + [p for t in tokens for p in _pobjs(t, *MEMBER_PREPS)]


def _asset_owners(tokens: list[Token], index: dict[int, _Mention]) -> list[Token]:
    """Targets plus the owner of a targeted military asset: "attacks on US bases" -> US. The owner must be an
    organisation, an abbreviated place or a possessive; "bases in Bahrain" never makes Bahrain the target."""
    assets = [t for t in tokens if t.i not in index and (t.lemma_.lower() in MILITARY_ASSETS
                                                         or t.lower_.removesuffix("s") in MILITARY_ASSETS)]
    owners = [c for t in assets for left in t.lefts for c in left.subtree  # "the US Navy warship": US
              if c.dep_ in ("compound", "poss") and _actor_modifier(c, index)]
    return tokens + owners


def _money(tok: Token) -> bool:
    return any(t.ent_type_ == "MONEY" or t.lemma_.lower() in MONEY_WORDS for t in tok.subtree)


Rule = tuple[str, str, list, list]


def _meeting_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """met_with (symmetric): A met B, A and B met, A held talks with B, a meeting between A and B / A's meeting with B.

    Triggers: meet verbs (active, passive, joint subject), "A hosted B" only when B is a person (a country hosting a
    base or a group is not a meeting), hold/have/conduct + meeting/talks/summit, and the nouns meeting/talks/summit
    with "between" or a possessive owner plus "with". Planned, cancelled, denied and negated meetings are
    suppressed. Known failures: indirect attribution, a meeting reported only as "the two met".
    """
    lemma, verb, noun = tok.lemma_.lower(), tok.pos_ in ("VERB", "AUX"), tok.pos_ == "NOUN"
    if verb and lemma in MEET_VERBS and not _suppressed(tok):
        if passive := _passive_subjects(tok):
            guests = [m for m in _ents(passive, index) if lemma != "host" or m.entity_type == "PERSON"]
            yield "met_with", "met_with.passive", guests, _ents(_agents(tok), index)
        elif lemma == "host":
            # Hosting a visitor is a meeting; a country hosting a base, a command or a group is not ("Qatar hosts
            # Hamas", "Bahrain, which hosts CENTCOM"), so only a person can be the hosted party.
            guests = [m for m in _ents(_children(tok, "dobj"), index) if m.entity_type == "PERSON"]
            yield "met_with", "met_with.hosted", _ents(_subjects(tok), index), guests
        else:
            subjects = _ents(_subjects(tok), index)
            objects = _ents(_children(tok, "dobj") + _pobjs(tok, "with"), index)
            yield ("met_with", "met_with.verb", subjects, objects) if objects else (
                "met_with", "met_with.joint_subject", subjects, subjects)
    if verb and lemma in HOLD_VERBS and not _suppressed(tok):
        for obj in _children(tok, "dobj"):
            if obj.lemma_.lower() in MEET_NOUNS and not _blocked(obj):  # "held no talks"
                subjects = _ents(_subjects(tok), index)
                others = _ents(_pobjs(obj, "with") + _pobjs(tok, "with"), index)
                yield "met_with", "met_with.held_talks", subjects, others or subjects
    if noun and lemma in MEET_NOUNS and not _suppressed(tok):
        between = _ents(_pobjs(tok, "between"), index)
        yield "met_with", "met_with.meeting_between", between, between
        owners, partners = _ents(_children(tok, "poss"), index), _ents(_pobjs(tok, "with"), index)
        yield "met_with", "met_with.meeting_with", owners, partners


def _criticism_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """criticized (A -> B): A criticized, condemned, denounced, accused or blamed B, or an action attributed to B.

    Triggers: CRITICIZE_VERBS with an entity subject and entity object, the passive with a by-agent, and the actor
    behind a criticized action ("condemned Israel's strikes", "condemned US sanctions"). Known failures: quoted
    allegations, irony, and criticism expressed without one of these verbs.
    """
    if not (tok.pos_ in ("VERB", "AUX") and tok.lemma_.lower() in CRITICIZE_VERBS) or _suppressed(tok):
        return
    if passive := _passive_subjects(tok):
        yield "criticized", "criticized.passive", _ents(_agents(tok), index), _ents(passive, index)
        return
    subjects, objects = _ents(_subjects(tok), index), _children(tok, "dobj")
    yield "criticized", "criticized.active", subjects, _ents(objects, index)
    # The actor behind a criticized action: "Israel's strikes", "the attack by Israel", "the Hamas attack", "US
    # sanctions". A title-case place modifier is left out: "Gaza attack" and "Iran sanctions" name a place or a
    # target, not the actor.
    owners = [p for o in objects if o.i not in index or index[o.i].entity_type == "TOPIC"
              for p in _event_actors(o, index)]
    yield "criticized", "criticized.object_possessor", subjects, _ents(owners, index)


def _topic_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """discussed_topic (A -> TOPIC B): A discussed, negotiated, debated, warned about, commented on, spoke or testified
    about, or announced something named by topic B.

    Triggers: DISCUSS_VERBS whose direct object (or passive subject) contains the topic phrase, or whose about/on/
    over/regarding complement does; the nouns talks/negotiations/discussion/debate/warning with a topic complement
    or compound ("nuclear talks"), owned by a possessive, a with/between participant, or the subject of the verb
    that governs them ("Iran held talks on ..."). Only explicit topic phrases from topics.toml count: the general
    subject of an article is never a discussed topic. Known failures: implicit subjects ("the talks focused on ...")
    and paraphrased topics outside the vocabulary.
    """
    lemma, verb, noun = tok.lemma_.lower(), tok.pos_ in ("VERB", "AUX"), tok.pos_ == "NOUN"
    if verb and lemma in DISCUSS_VERBS and not _suppressed(tok):
        objects = [] if lemma == "talk" else _children(tok, "dobj", "nsubjpass")
        preps = [c for c in tok.children if c.dep_ == "prep" and c.lower_ in TOPIC_PREPS]
        if lemma == "announce":
            # Only what was announced: not a reason or a target ("announced sanctions on Iran over its nuclear
            # programme"), and not sanctions or strikes, which sanctioned/attacked already type.
            preps = []
            objects = [o for o in objects if not _sanction_noun(o) and o.lemma_.lower() not in ATTACK_NOUNS]
        actors = _subjects(tok) + _agents(tok) + _pobjs(tok, "with") + [p for o in objects for p in _pobjs(o, "with")]
        if tok.dep_ == "advcl" and _infinitive(tok) and tok.head.lemma_.lower() in MEET_VERBS:
            actors += _children(tok.head, "dobj") + _pobjs(tok.head, "with")  # "X met Y ... to discuss Z"
        topics = _region_topics(objects + preps, index)
        yield "discussed_topic", "discussed_topic.verb", _ents(actors, index), topics
    if noun and lemma in DISCUSS_NOUNS and not _suppressed(tok):
        preps = [c for c in tok.children if c.dep_ == "prep" and c.lower_ in TOPIC_PREPS]
        actors = _children(tok, "poss") + _pobjs(tok, "with", "between")
        if tok.dep_ == "dobj" and tok.head.pos_ == "VERB":
            # "Iran held talks on ...", or the owner of a non-entity subject: "Anthropic's IPO pitch includes a
            # warning about human extinction".
            actors += [p for s in _subjects(tok.head) for p in ([s] if s.i in index else _children(s, "poss"))]
        topics = _region_topics(preps + _children(tok, "compound"), index, shallow=[tok])
        yield "discussed_topic", "discussed_topic.noun", _ents(actors, index), topics


def _role_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """Copular roles and leadership/founding/employment verbs.

    affiliated_with (A -> ORG B): "A is the CEO of B" (present-tense copula + ROLE_WORDS), a person "leads/heads/
    chairs/runs" B, "A founded B" / "B was founded by A", "A works at/for B" (not "worked at"). invested_in and
    partnered_with have copular forms too: "A is B's biggest investor", "A is B's partner". Former roles ("former
    head", "was the CEO") are never current affiliations. Known failures: roles stated without an organisation,
    organisations named only by a pronoun, and a past role written in the present ("X, OpenAI's co-founder").
    """
    lemma, verb = tok.lemma_.lower(), tok.pos_ in ("VERB", "AUX")
    # Past tense describes a role that may have ended ("John Sculley was the CEO of Apple", "worked at OpenAI until
    # 2024"), so copular and work-verb forms need a present (or present perfect) verb.
    past = tok.tag_ == "VBD"
    if lemma == "be" and not past and not _suppressed(tok):
        for attr in _children(tok, "attr"):
            role = attr.lemma_.lower()
            if attr.i in index:
                continue
            subjects = _ents(_subjects(tok), index)
            if role in ROLE_WORDS:
                yield "affiliated_with", "affiliated_with.copular", subjects, _role_orgs(attr, index)
            elif role in INVESTOR_WORDS:
                yield "invested_in", "invested_in.copular", subjects, _role_orgs(attr, index, ("in", "of"))
            elif role == "partner":  # "A is B's partner"; "A and B are partners"
                yield "partnered_with", "partnered_with.copular", subjects, _role_orgs(attr, index, ("of", "to")) or (
                    subjects if attr.tag_ == "NNS" else [])
    if not verb or _suppressed(tok):
        return
    if lemma in LEAD_VERBS | PERSON_LEAD_VERBS:
        # A person leads, heads, chairs or runs B; an organisation "leading" another is usually a race or a market
        # ("OpenAI leads Google in the race to build agents").
        people = [m for m in _ents(_subjects(tok), index) if m.entity_type == "PERSON"]
        rule = "affiliated_with.runs" if lemma in PERSON_LEAD_VERBS else "affiliated_with.leads"
        yield "affiliated_with", rule, people, _ents(_children(tok, "dobj"), index)
    if lemma in FOUND_VERBS:
        if passive := _passive_subjects(tok):
            yield "affiliated_with", "affiliated_with.founded", _ents(_agents(tok), index), _ents(passive, index)
        else:
            yield "affiliated_with", "affiliated_with.founded", _ents(_subjects(tok), index), _ents(
                _children(tok, "dobj"), index)
    if lemma in WORK_VERBS and not past:
        people = [m for m in _ents(_subjects(tok), index) if m.entity_type == "PERSON"]
        yield "affiliated_with", "affiliated_with.works_for", people, _ents(_pobjs(tok, "at", "for"), index)


def _acquisition_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """acquired (PERSON/ORG A -> ORG B): A bought or acquired organisation B, or signed an agreement to.

    Triggers: acquire/buy/purchase with an entity subject and entity object ("Nvidia acquired Hugging Face"), the
    passive ("Hugging Face was acquired by Nvidia", also in a relative clause), and the nouns acquisition/purchase/
    takeover/buyout with a possessive, compound or by-agent acquirer and an "of" object ("Microsoft's acquisition of
    Activision"). "agrees to acquire" and "a deal to buy" count as an announced deal (ANNOUNCED_DEAL_HEADS); plans,
    talks, bids, offers, attempts, modals, negation and questions are suppressed. Known failures: a deal that later
    collapsed is still an edge, and "buy" for buying a company's products ("Meta buys Nvidia") reads as an
    acquisition.
    """
    lemma = tok.lemma_.lower()
    if tok.pos_ == "VERB" and lemma in ACQUIRE_VERBS and not _suppressed(tok, ANNOUNCED_DEAL_HEADS):
        if passive := _passive_subjects(tok):
            yield "acquired", "acquired.passive", _ents(_agents(tok), index), _ents(passive, index)
        else:
            yield "acquired", "acquired.verb", _ents(_subjects(tok), index), _ents(_children(tok, "dobj"), index)
    if tok.pos_ == "NOUN" and lemma in ACQUIRE_NOUNS and not _suppressed(tok, ANNOUNCED_DEAL_HEADS):
        yield "acquired", "acquired.noun", _ents(_event_actors(tok, index), index), _ents(_pobjs(tok, "of"), index)


def _investment_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """invested_in (PERSON/ORG A -> ORG B): A put money into B.

    Triggers: "A invested ... in/into B"; the nouns investment/stake with an "in" object and an owner that is a
    possessive, compound or by-agent, or the subject of a take/buy/make/announce verb governing them ("SoftBank
    took a stake in OpenAI", "Microsoft's investment in OpenAI"); "B raised $X / a round from A" (a money object is
    required); "B raised ... in a round led by A". The copular, appositive and title forms ("A is B's biggest
    investor", "A, an investor in B", "B investor A") live with the role rules. Rejected, turned-down, planned and
    negated investments are suppressed. Known failures: investment announced as a plan in the future tense is
    dropped; the investor of an unnamed company ("raised money from Nvidia") is lost.
    """
    lemma, verb = tok.lemma_.lower(), tok.pos_ == "VERB"
    if verb and lemma in INVEST_VERBS and not _suppressed(tok):
        # "invested $10bn in OpenAI": the parser may hang "in OpenAI" on the amount
        targets = _pobjs(tok, "in", "into") + [p for o in _children(tok, "dobj") for p in _pobjs(o, "in", "into")]
        yield "invested_in", "invested_in.verb", _ents(_subjects(tok), index), _ents(targets, index)
    if tok.pos_ == "NOUN" and lemma in INVEST_NOUNS and not _suppressed(tok):
        actors = _event_actors(tok, index)
        targets = _pobjs(tok, "in", "into")
        head = tok.head
        if tok.dep_ == "dobj" and head.pos_ == "VERB" and head.lemma_.lower() in STAKE_VERBS | {
                "make", "announce", "complete"}:
            actors += _subjects(head)
            targets += _pobjs(head, "in", "into")
        yield "invested_in", "invested_in.noun", _ents(actors, index), _ents(targets, index)
    if verb and lemma in RAISE_VERBS and not _suppressed(tok) and any(_money(o) for o in _children(tok, "dobj")):
        companies = _ents(_subjects(tok), index)
        yield "invested_in", "invested_in.raised_from", _ents(_pobjs(tok, "from"), index), companies
        leads = [t for t in tok.subtree if t.lemma_.lower() == "lead" and t.dep_ == "acl" and t.i != tok.i]
        yield "invested_in", "invested_in.round_led_by", _ents([a for t in leads for a in _agents(t)], index), companies


def _release_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """released (ORG A -> ORG B): A released, launched, unveiled, shipped or announced product B.

    Triggers: RELEASE_VERBS ("roll" only with "out") whose direct object is a product noun (model, chatbot, app,
    chip, tool, ...) modified by the product's name: "Google announces Gemini 4 Argon AI model", "Meta released its
    new Llama model". A bare entity object is not enough, because "released" also means freed ("released the Galaxy
    Leader") and "launched" takes operations and campaigns. Planned, negated and modal releases are suppressed.
    Known failures: a product named without a product noun ("Meta released Llama 4") is missed, and an organisation
    named as the modifier ("released an OpenAI model") reads as the product.
    """
    lemma = tok.lemma_.lower()
    if tok.pos_ != "VERB" or lemma not in RELEASE_VERBS or (lemma == "roll" and not _has_particle(tok, "out")):
        return
    if _suppressed(tok):
        return
    objects = [o for o in _children(tok, "dobj") + _passive_subjects(tok) if o.lemma_.lower() in PRODUCT_NOUNS
               and (o.i not in index or index[o.i].entity_type == "TOPIC")]  # "AI model" is also a topic phrase
    products = [c for o in objects for c in o.children if c.dep_ in ("compound", "nmod", "amod") and c.i in index]
    makers = _agents(tok) if _passive_subjects(tok) else _subjects(tok)
    yield "released", "released.product", [m for m in _ents(makers, index) if m.entity_type == "ORG"], _ents(
        products, index)


def _partnership_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """partnered_with (symmetric, ORG-ORG): A and B partnered, teamed up, or made a deal with each other.

    Triggers: partner/collaborate verbs and "team up" with "with" or a joint subject; sign/strike/reach/ink/seal/
    announce/form/enter/have/agree + partnership/deal/agreement/collaboration/alliance/pact/contract with a "with"
    participant or a joint subject; the same nouns with "between", or a possessive/compound owner plus "with"
    ("Microsoft's partnership with OpenAI"). The appositive/copular "A, B's partner" forms live with the role rules.
    Negated ("don't have a deal with"), planned and sought deals are suppressed. Known failures: an acquisition
    agreement also reads as a deal between the two companies, and a settlement "agreement with" a regulator reads as
    a partnership.
    """
    lemma, verb = tok.lemma_.lower(), tok.pos_ == "VERB"
    if verb and (lemma in PARTNER_VERBS or (lemma == "team" and _has_particle(tok, "up"))) and not _suppressed(tok):
        subjects, partners = _ents(_subjects(tok), index), _ents(_pobjs(tok, "with"), index)
        yield ("partnered_with", "partnered_with.verb", subjects, partners) if partners else (
            "partnered_with", "partnered_with.joint_subject", subjects, subjects)
    if verb and lemma in DEAL_VERBS and not _suppressed(tok):
        for obj in _children(tok, "dobj"):
            if obj.lemma_.lower() in DEAL_NOUNS and not _blocked(obj):
                subjects = _ents(_subjects(tok), index)
                others = _ents(_pobjs(obj, "with") + _pobjs(tok, "with"), index)
                yield "partnered_with", "partnered_with.deal", subjects, others or subjects
    if tok.pos_ == "NOUN" and lemma in DEAL_NOUNS and not _suppressed(tok):
        between = _ents(_pobjs(tok, "between"), index)
        yield "partnered_with", "partnered_with.deal_between", between, between
        owners = _children(tok, "poss") + [c for c in _children(tok, "compound") if _actor_modifier(c, index)]
        yield "partnered_with", "partnered_with.deal_with", _ents(owners, index), _ents(_pobjs(tok, "with"), index)


def _sanction_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """sanctioned (A -> B): A imposed sanctions or an embargo on B.

    Triggers: sanction/blacklist verbs (active, passive, relative clause "which is sanctioned by A"); impose/
    reimpose/announce/unleash/slap/levy/place/issue/introduce/adopt/expand/extend/tighten + sanctions/embargo with
    an on/against target or a title-case place modifier ("issued new Iran sanctions"); the noun with an actor
    modifier and an on/against target ("EU sanctions against Iran", "the UK's sanctions regime against Iran"); "B
    is under A sanctions". Lifted, waived, threatened, sought, proposed and negated sanctions are suppressed.
    Known failures: a relative clause attached to the wrong noun by the parser, and targets named only through a
    list ("individuals including X").
    """
    lemma = tok.lemma_.lower()
    if tok.pos_ == "VERB" and lemma in SANCTION_VERBS and not _suppressed(tok):
        if passive := _passive_subjects(tok):
            yield "sanctioned", "sanctioned.passive", _ents(_agents(tok), index), _ents(passive, index)
        else:
            objects = _children(tok, "dobj")
            yield "sanctioned", "sanctioned.verb", _ents(_subjects(tok), index), _ents(
                _members(objects) + _pobjs(tok, *MEMBER_PREPS), index)
    if tok.pos_ == "VERB" and lemma in IMPOSE_VERBS and not _suppressed(tok):
        passive = _passive_subjects(tok)
        for noun in (n for n in _event_objects(tok) + passive if _sanction_noun(n) and not _blocked(n)):
            actors = _agents(tok) + _event_actors(noun, index) if noun in passive else _subjects(tok)
            targets = (_members(_pobjs(noun, *EVENT_TARGET_PREPS) + _pobjs(tok, *EVENT_TARGET_PREPS))
                       + _target_places(noun, index))
            yield "sanctioned", "sanctioned.imposed", _ents(actors, index), _ents(targets, index)
    if _sanction_noun(tok) and not _suppressed(tok) and tok.head.lemma_.lower() not in LIFT_VERBS:
        targets = _pobjs(tok, *EVENT_TARGET_PREPS) + _target_places(tok, index)
        under = tok.head
        if tok.dep_ == "pobj" and under.lower_ == "under" and under.head.pos_ in ("VERB", "AUX"):
            targets += _subjects(under.head)  # "Iran has been under US sanctions"
        yield "sanctioned", "sanctioned.noun", _ents(_event_actors(tok, index), index), _ents(targets, index)


def _attack_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """attacked (ORG/LOCATION A -> ORG/LOCATION B): A carried out a military or cyber attack on B.

    Triggers: attack/strike/bomb/invade/shell/hack verbs (active, passive); launch/carry out/conduct/stage/mount/
    unleash/wage + attack nouns with an on/against target ("launched dozens of strikes on Iran", "waged war on");
    the nouns attack/strike/airstrike/bombing/assault/offensive/raid/invasion/war/bombardment/counterattack/
    cyberattack with an actor (possessive, by-agent, organisation or abbreviated place modifier) and an on/against
    target, or an "of" target for invasion/bombing/bombardment; a nationality adjective ("Iranian attacks on
    Bahrain") or, with an explicit target, a place compound ("Israel strikes on Gaza") names the attacker. People are
    excluded at both ends and one side must be a place, because a person or a company "attacking" another is usually
    verbal ("OpenAI attacked Anthropic in a blog post"); "attacked B over/for X" is criticism. Attempted,
    threatened, deterred, planned, hypothetical ("any strike would ..."), claimed ("Iran claimed Israel attacked")
    and negated attacks are suppressed. Known failures: a labour "strike against" a place-named organisation, and
    attacks reported without attribution are taken as stated.
    """
    lemma = tok.lemma_.lower()
    # "China attacked the US over its tariffs", "attacked Meta for its data policy": criticism, not an attack.
    if tok.pos_ == "VERB" and lemma in ATTACK_VERBS and not _suppressed(tok) and not _pobjs(tok, "over", "for"):
        if passive := _passive_subjects(tok):
            yield "attacked", "attacked.passive", _ents(_agents(tok), index), _ents(passive, index)
        else:
            yield "attacked", "attacked.verb", _ents(_subjects(tok), index), _ents(
                _asset_owners(_children(tok, "dobj"), index), index)
    if (tok.pos_ == "VERB" and lemma in LAUNCH_VERBS and (lemma != "carry" or _has_particle(tok, "out"))
            and not _suppressed(tok)):
        for noun in _event_objects(tok):
            if noun.lemma_.lower() in ATTACK_NOUNS and not _blocked(noun):
                targets = _asset_owners(_pobjs(noun, *EVENT_TARGET_PREPS) + _pobjs(tok, *EVENT_TARGET_PREPS), index)
                yield "attacked", "attacked.launched", _ents(_subjects(tok), index), _ents(targets, index)
    if tok.pos_ == "NOUN" and lemma in ATTACK_NOUNS and not _suppressed(tok):
        preps = EVENT_TARGET_PREPS + (("of",) if lemma in OF_TARGET_NOUNS else ())
        yield "attacked", "attacked.noun", _ents(_event_actors(tok, index), index), _ents(
            _asset_owners(_pobjs(tok, *preps), index), index)


def _news_outlet(mention: _Mention) -> bool:
    name = normalize_name(mention.canonical or "")
    words = re.split(r"[\s'’-]+", name)
    return mention.entity_type == "ORG" and (name in NEWS_NAMES or not NEWS_WORDS.isdisjoint(words))


def _quote_rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    """quoted_by (PERSON/ORG A -> news outlet B): B reports A's words.

    Triggers: "A told B", including "A told B's reporter X" (the reporter's outlet); "A spoke to/with B"; "A
    confirmed to B" (also "to the B news agency"); "A said in an interview with B"; "A was quoted/cited by B". A
    speaker named only by a role ("a Pentagon spokesperson") is the role's organisation. B must be an organisation
    whose name marks it as a news outlet (NEWS_WORDS, NEWS_NAMES), so "A told the UN Security Council" or "told the
    IAEA" is not a quote, and "told B to ..." is an order. Refusals ("declined to tell Reuters") and negation are
    suppressed. Known failures: "according to B" without a named speaker is not used, and an outlet whose name has no
    news word (a newspaper called only by its city) is missed.
    """
    lemma = tok.lemma_.lower()
    if tok.pos_ != "VERB" or lemma not in QUOTE_VERBS or _suppressed(tok):
        return
    if lemma == "tell" and any(_infinitive(c) for c in _children(tok, "xcomp", "ccomp")):
        return  # "Hegseth told the agency to stand down" is an order, not a quote
    if passive := _passive_subjects(tok):
        if lemma in ("quote", "cite"):
            yield "quoted_by", "quoted_by.passive", _ents(passive, index), [
                m for m in _ents(_agents(tok), index) if _news_outlet(m)]
        return
    if lemma == "tell":
        listeners = _children(tok, "dobj", "dative")
        outlets = _ents(listeners, index) + [index[x.i] for t in listeners for c in _children(t, "poss")
                                             for x in c.subtree if x.i in index]
    elif lemma in ("speak", "confirm"):
        # "confirmed to the Reuters news agency": an outlet named only as the compound of agency/outlet/network.
        listeners = _pobjs(tok, "to") if lemma == "confirm" else _pobjs(tok, "to", "with")
        outlets = _ents(listeners, index) + [index[c.i] for t in listeners if t.i not in index
                                             and t.lemma_.lower() in OUTLET_NOUNS
                                             for c in _children(t, "compound") if c.i in index]
    elif lemma == "say":
        talks = [n for n in _pobjs(tok, "in", "during") if n.lemma_.lower() in INTERVIEW_NOUNS]
        outlets = _ents([p for n in talks for p in _pobjs(n, "with")] + [c for n in talks
                                                                           for c in _children(n, "compound", "poss")],
                        index)
    else:
        return
    # A speaker named by a role ("a Pentagon spokesperson told ...") is the organisation the role belongs to.
    speakers = _ents(_subjects(tok), index) or [m for s in _subjects(tok) if s.lemma_.lower() in SPOKES_WORDS
                                                for m in _role_orgs(s, index)]
    yield "quoted_by", f"quoted_by.{lemma}", speakers, [m for m in outlets if _news_outlet(m)]


RULES = (_meeting_rules, _criticism_rules, _topic_rules, _role_rules, _acquisition_rules, _investment_rules,
         _release_rules, _partnership_rules, _sanction_rules, _attack_rules, _quote_rules)


def _rules(tok: Token, index: dict[int, _Mention]) -> Iterator[Rule]:
    for rule in RULES:
        yield from rule(tok, index)


def _joining_hyphen(tok: Token) -> bool:
    return tok.text == "-" and not tok.whitespace_ and tok.i > 0 and not tok.doc[tok.i - 1].whitespace_


def _role_vocabulary(tok: Token) -> bool:
    word = tok.lower_.rstrip(".")
    return (tok.lemma_.lower() in ROLE_WORDS | INVESTOR_WORDS or word in TITLES or word in ROLE_PREFIXES
            or _joining_hyphen(tok))


def _affiliations(sent: Span, index: dict[int, _Mention]) -> Iterator[Rule]:
    """Noun-phrase roles: titles, possessives and appositives (no verb involved).

    affiliated_with: "NATO Secretary General Mark Rutte", "OpenAI CEO Sam Altman", "Nvidia founder Jensen Huang"
    (affiliated_with.title); "Anthropic's Dario Amodei" (affiliated_with.possessive: an organisation's possessive
    directly on a person's name); "Sam Altman, OpenAI's chief executive" (affiliated_with.appositive). With an
    investor noun the same shapes give invested_in ("OpenAI investor SoftBank", "SoftBank, an investor in OpenAI");
    "Microsoft, OpenAI's partner" gives partnered_with. Former roles never count.
    """
    doc = sent.doc
    actors = list({id(m): m for m in index.values() if m.entity_type in ("PERSON", "ORG")}.values())
    for org in (m for m in actors if m.entity_type == "ORG"):
        for m in actors:
            if not 0 < m.span.start - org.span.end <= 5:
                continue
            between = doc[org.span.end:m.span.start]
            # "Former NATO chief X", "ex-Google engineer X": the modifier may precede the organisation or hang off the
            # role word.
            former = _former_prefix(org.span) or any(c.lower_ in FORMER for t in between for c in t.children)
            if former or any((t.is_punct and not _joining_hyphen(t)) or t.pos_ == "VERB" or t.lower_ in FORMER
                             or t.i in index for t in between):
                continue
            # "NATO Secretary General Mark Rutte", "UN Secretary-General Guterres", "the EU's foreign policy chief".
            # The parse must link the two, unless every word between is role vocabulary ("OpenAI co-founder X").
            words = [t for t in between if t.tag_ != "POS"]
            linked = m.span.root.is_ancestor(org.span.root) or (words and all(_role_vocabulary(t) for t in words))
            if linked and any(t.lemma_.lower() in ROLE_WORDS for t in between):
                yield "affiliated_with", "affiliated_with.title", [m], [org]
            elif linked and any(t.lemma_.lower() in INVESTOR_WORDS for t in between):
                yield "invested_in", "invested_in.title", [m], [org]
            # "Anthropic's Dario Amodei": nothing but the possessive marker between an organisation and a person.
            elif (m.entity_type == "PERSON" and len(between) == 1 and between[0].tag_ == "POS"
                  and org.span.root.dep_ == "poss" and org.span.root.head.i in range(m.span.start, m.span.end)):
                yield "affiliated_with", "affiliated_with.possessive", [m], [org]
    for m in actors:
        root = m.span.root
        if m.entity_type == "PERSON":  # "Muhanad Seloom of the Doha Institute"
            yield "affiliated_with", "affiliated_with.of", [m], [o for o in _ents(_pobjs(root, "of"), index)
                                                                 if o.entity_type == "ORG"]
        roles = _children(root, "appos") + ([root.head] if root.dep_ == "appos" else [])
        for role in roles:
            if role.i in index:
                continue
            word = role.lemma_.lower()
            # People only: in headline style "Nvidia, Supermicro employees charged ..." the comma means "and",
            # and an organisation is not a role-holder of another organisation (edge #860 in the README).
            if word in ROLE_WORDS and m.entity_type == "PERSON":
                yield "affiliated_with", "affiliated_with.appositive", [m], _role_orgs(role, index)
            elif word in INVESTOR_WORDS:
                yield "invested_in", "invested_in.appositive", [m], _role_orgs(role, index, ("in", "of"))
            elif word == "partner":
                yield "partnered_with", "partnered_with.appositive", [m], _role_orgs(role, index, ("of", "to"))


def _eligible(relation: str, source: _Mention, target: _Mention) -> bool:
    if source.key == target.key:
        return False
    if relation == "released":
        return source.entity_type == target.entity_type == "ORG"
    if relation in ("affiliated_with", "acquired", "invested_in", "quoted_by"):
        return source.entity_type in ("PERSON", "ORG") and target.entity_type == "ORG"
    if relation == "partnered_with":
        return source.entity_type == target.entity_type == "ORG"
    if relation == "attacked":
        # One side must be a place: "OpenAI attacked Anthropic in a blog post" is verbal, "the IRGC attacked a US
        # vessel" and "Israel's strikes on Hezbollah" are not.
        types = {source.entity_type, target.entity_type}
        return types <= {"ORG", "LOCATION"} and "LOCATION" in types
    if relation == "discussed_topic":
        return source.entity_type != "TOPIC" and target.entity_type == "TOPIC"
    return "TOPIC" not in (source.entity_type, target.entity_type)


def _window(sent: Span, *mentions: _Mention) -> tuple[int, int]:
    text, start, end = sent.doc.text, sent.start_char, sent.end_char
    if end - start > MAX_EVIDENCE_CHARS:
        low, high = min(m.start for m in mentions), max(m.end for m in mentions)
        pad = max(0, MAX_EVIDENCE_CHARS - (high - low)) // 2
        start, end = max(start, low - pad), min(end, high + pad)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _relation(segment: Segment, sent: Span, relation: str, rule: str, a: _Mention, b: _Mention) -> Relation:
    directed = relation in DIRECTED
    if not directed and b.key < a.key:
        a, b = b, a
    start, end = _window(sent, a, b)
    return Relation(source_key=a.key, target_key=b.key, relation_type=relation, directed=directed,
                    evidence_text=segment.text[start:end], segment_id=segment.id, start=start, end=end,
                    rule_id=rule, quality_tier="cooccurrence" if relation == "mentioned_with" else "semantic_rule")


def _relations(segment: Segment, sent: Span, index: dict[int, _Mention], warnings: list[str]) -> list[Relation]:
    """Relations are only ever built between entities inside this one sentence of this one segment."""
    local = {i: m for i, m in index.items() if sent.start <= m.span.start and m.span.end <= sent.end}
    last = next((t for t in reversed(sent) if not (t.is_space or t.is_quote or t.text in ")]")), None)
    # A question ("Did Biden really meet Xi?") asserts no event; only weak co-occurrence is kept.
    asks = last is not None and "?" in last.text
    rules = [] if asks else [r for tok in sent for r in _rules(tok, local)] + list(_affiliations(sent, local))
    result = [_relation(segment, sent, relation, rule, s, t) for relation, rule, sources, targets in rules
              for s in sources for t in targets if _eligible(relation, s, t)]
    linked = {frozenset((r.source_key, r.target_key)) for r in result}
    entities: dict[str, _Mention] = {}
    for m in sorted(local.values(), key=lambda m: m.start):
        entities.setdefault(m.key, m)
    if len(entities) > MAX_SENTENCE_ENTITIES:
        warnings.append(f"Co-occurrence skipped for a sentence with {len(entities)} entities in segment {segment.id}")
        return result
    for a, b in combinations(entities.values(), 2):
        if frozenset((a.key, b.key)) not in linked and not (a.entity_type == b.entity_type == "TOPIC"):
            result.append(_relation(segment, sent, "mentioned_with", "mentioned_with.sentence", a, b))
    return result
