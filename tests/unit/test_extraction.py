"""Extraction rules against the real local en_core_web_sm model.

Every sentence below is labelled synthetic: written for these tests in the style of geopolitical news, AI-industry
news, Hacker News, Mastodon, Reddit, and X text. None of it is crawled evidence, and the events it describes (deals,
investments, sanctions, attacks) are not claims about the real world. The exception is the "calibration corpus"
section: those sentences are quoted from pages crawled on 7 October 2026 (data/calibration.sqlite3) to pin the
precision fixes made after reviewing the real output.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import spacy

from media_intelligence.extraction import DIRECTED, RELATION_DEFINITIONS, Extractor, load_aliases, load_topics
from media_intelligence.models import ContentItem, Segment

CONFIG = Path(__file__).resolve().parents[2] / "config"


@pytest.fixture(scope="module")
def extractor():
    return Extractor("en_core_web_sm", CONFIG / "aliases.toml", CONFIG / "topics.toml")


@pytest.fixture(scope="module")
def run(extractor):
    def run_texts(*texts: str):
        segments = [Segment(id=f"s{i}", text=text) for i, text in enumerate(texts)]
        item = ContentItem(source_url="https://example.com/synthetic", source_type="news",
                           scraped_at=datetime(2026, 10, 7, tzinfo=UTC), body="\n\n".join(texts), segments=segments)
        result = extractor.extract(item)
        check_invariants(segments, result)
        return result
    return run_texts


def check_invariants(segments, result):
    text = {s.id: s.text for s in segments}
    for m in result.mentions:
        assert text[m.segment_id][m.start:m.end] == m.name
        assert m.resolution in {"exact", "alias", "surname", "handle", "abbreviation"}
    for r in result.relations:
        assert r.source_key != r.target_key
        assert text[r.segment_id][r.start:r.end] == r.evidence_text and r.evidence_text == r.evidence_text.strip()
        assert r.directed or r.source_key <= r.target_key
        assert (r.quality_tier == "cooccurrence") == (r.relation_type == "mentioned_with")
        assert r.rule_id.startswith(r.relation_type)
    keys = [(r.source_key, r.target_key, r.relation_type) for r in result.relations]
    assert len(keys) == len(set(keys))


def keys(result):
    return {m.key for m in result.mentions}


def semantic(result):
    return {(r.source_key, r.relation_type, r.target_key) for r in result.relations
            if r.quality_tier == "semantic_rule"}


def relation_types(result):
    return {r.relation_type for r in result.relations}


# Entities and topics

def test_entity_type_mapping_and_norp_exclusion(run):
    result = run("US President Joe Biden met with Chinese President Xi Jinping in Beijing on Tuesday. "
                 "Mark Rutte, the head of NATO, criticized Hungary.")
    types = {m.key: m.entity_type for m in result.mentions}
    assert types["PERSON:joe biden"] == "PERSON" and types["ORG:nato"] == "ORG"
    assert types["LOCATION:beijing"] == types["LOCATION:united states"] == "LOCATION"
    assert not any("chinese" in key for key in types)  # NORP nationality is not an organisation


def test_empty_text_without_entities(run):
    result = run("It was a quiet day with nothing to report.")
    assert result.mentions == [] and result.relations == []


def test_item_without_segments_uses_body(extractor):
    item = ContentItem(source_url="https://example.com/synthetic", source_type="news",
                       scraped_at=datetime(2026, 10, 7, tzinfo=UTC), body="Antony Blinken met Benjamin Netanyahu.")
    result = extractor.extract(item)
    assert {m.segment_id for m in result.mentions} == {"body"}
    assert ("PERSON:antony blinken", "met_with", "PERSON:benjamin netanyahu") in semantic(result)


def test_topics_multiword_synonyms_and_longest_match(run):
    result = run("Iran defended its nuclear programme, the truce collapsed, and the trade war with China escalated.")
    found = {(m.name, m.key, m.resolution) for m in result.mentions if m.entity_type == "TOPIC"}
    assert ("nuclear programme", "TOPIC:nuclear programme", "exact") in found
    assert ("truce", "TOPIC:ceasefire", "alias") in found
    assert ("trade war", "TOPIC:trade", "alias") in found
    assert "TOPIC:military conflict" not in keys(result)  # "war" inside "trade war" is not a second topic


def test_topic_word_boundaries_and_verb_senses(run):
    result = run("The electionsoftware vendor denied the report. Russia and Ukraine continued to trade accusations. "
                 "Fans debated whether the trades were fair.")
    assert not {k for k in keys(result) if k.startswith("TOPIC:")}


def test_topic_expected_miss_and_known_overmatch(run):
    # "polls" is outside the vocabulary (expected miss); a sports "trade" noun still matches (documented limit).
    assert "TOPIC:elections" not in keys(run("New polls show voters are angry."))
    assert "TOPIC:trade" in keys(run("The Lakers completed a trade before the deadline."))


def test_junk_spans_handles_and_usernames_are_not_nodes(run):
    result = run("Thanks @SomeAnalyst and u/geo_nerd for the link https://example.com/story "
                 "about 2024 elections in Iran.")
    assert keys(result) == {"TOPIC:elections", "LOCATION:iran"}


def test_titles_are_stripped_from_person_names(extractor):
    doc = extractor.nlp("Prime Minister Keir Starmer spoke to Mr. Lammy.")
    doc.ents = [spacy.tokens.Span(doc, 0, 4, label="PERSON"), spacy.tokens.Span(doc, 6, 8, label="PERSON")]
    assert [m.span.text for m in extractor._candidates(doc)] == ["Keir Starmer", "Lammy"]


# Normalization and identity

def test_pdf_regression_name_surname_and_handle_share_one_node(run):
    result = run("Elon Musk met European officials in Brussels on Monday. Musk said the talks were useful.",
                 "@elonmusk later posted a video of the meeting.")
    musk = [(m.name, m.resolution) for m in result.mentions if m.key == "PERSON:elon musk"]
    assert musk == [("Elon Musk", "exact"), ("Musk", "alias"), ("@elonmusk", "handle")]
    assert not any(k.startswith("PERSON:") and k != "PERSON:elon musk" for k in keys(result))
    # The reviewed "Musk" alias also covers an item that never writes the full name.
    assert ("PERSON:elon musk", "criticized", "ORG:nato") in semantic(run("Musk criticized NATO over its spending."))


def test_unreviewed_surname_resolves_from_the_full_name_in_the_item(run):
    result = run("Mark Carney met Antony Blinken in Ottawa. Carney said the talks were useful.")
    assert [(m.name, m.resolution) for m in result.mentions if m.key == "PERSON:mark carney"] == [
        ("Mark Carney", "exact"), ("Carney", "surname")]


def test_two_people_sharing_a_surname_stay_unresolved(run):
    result = run("Elon Musk and his brother Kimbal Musk attended the launch. Musk said the rocket worked.")
    assert {m.name: m.key for m in result.mentions} == {"Elon Musk": "PERSON:elon musk",
                                                        "Kimbal Musk": "PERSON:kimbal musk"}
    assert any("Ambiguous surname 'Musk'" in w for w in result.warnings)


def test_single_word_alias_conflicting_with_item_context_is_ambiguous(run):
    result = run("Melania Trump visited Paris. Donald Trump met Emmanuel Macron. Trump said it went well.")
    assert [m.name for m in result.mentions if m.key == "PERSON:donald trump"] == ["Donald Trump"]
    assert any("Ambiguous surname 'Trump'" in w for w in result.warnings)


def test_reviewed_abbreviations_map_to_one_node(run):
    result = run("The UN and the U.S. discussed sanctions. The United Nations and the United States disagreed.",
                 "EU and NATO officials met. The European Union welcomed the IAEA report.")
    assert {"ORG:united nations", "LOCATION:united states", "ORG:european union", "ORG:nato",
            "ORG:international atomic energy agency"} <= keys(result)
    assert not {"ORG:un", "LOCATION:u.s.", "ORG:eu", "ORG:iaea"} & keys(result)


def test_pronoun_us_is_not_the_united_states(run):
    assert "LOCATION:united states" not in keys(run("Please join us in Geneva for the discussion."))


def test_unmapped_handle_never_becomes_a_person(run):
    result = run("Thanks to @ElonMusk and @SomeAnalyst for sharing this thread.")
    assert [(m.name, m.key) for m in result.mentions] == [("@ElonMusk", "PERSON:elon musk")]


def test_reviewed_alias_corrects_small_model_label(run):
    result = run("Recep Tayyip Erdogan met Vladimir Putin in Ankara.")
    assert ("PERSON:recep tayyip erdoğan", "met_with", "PERSON:vladimir putin") in semantic(result)


def test_alias_configuration_rejects_ambiguity():
    with pytest.raises(ValueError, match="more than one entity"):
        load_aliases({"entities": [{"canonical": "United States", "type": "LOCATION", "aliases": ["US"]},
                                   {"canonical": "Ultra Sonic", "type": "ORG", "aliases": ["us"]}]})
    with pytest.raises(ValueError):
        load_aliases({"entities": [{"canonical": "Diplomacy", "type": "TOPIC"}]})
    with pytest.raises(ValueError, match="more than one topic"):
        load_topics({"topics": [{"name": "trade", "patterns": ["tariffs"]},
                                {"name": "sanctions", "patterns": ["Tariffs"]}]})


def test_version_names_model_and_changes_with_configuration(extractor, tmp_path):
    assert "en_core_web_sm-3.8.0" in extractor.version
    aliases = tmp_path / "aliases.toml"
    extra = '\n[[entities]]\ncanonical = "Kaja Kallas"\ntype = "PERSON"\n'
    aliases.write_text((CONFIG / "aliases.toml").read_text() + extra)
    other = Extractor("en_core_web_sm", aliases, CONFIG / "topics.toml")
    assert other.version != extractor.version


# Relations

def test_met_with_positive_variants(run):
    result = run("Antony Blinken met Benjamin Netanyahu in Jerusalem.",
                 "Joe Biden met with Xi Jinping on the sidelines of the summit.",
                 "Emmanuel Macron and Olaf Scholz met in Berlin on Friday.",
                 "Prime Minister Narendra Modi held talks with Russian President Vladimir Putin.",
                 "A meeting between Antony Blinken and Wang Yi took place in Munich.")
    met = {(r.source_key, r.target_key, r.rule_id) for r in result.relations if r.relation_type == "met_with"}
    assert met == {("PERSON:antony blinken", "PERSON:benjamin netanyahu", "met_with.verb"),
                   ("PERSON:joe biden", "PERSON:xi jinping", "met_with.verb"),
                   ("PERSON:emmanuel macron", "PERSON:olaf scholz", "met_with.joint_subject"),
                   ("PERSON:narendra modi", "PERSON:vladimir putin", "met_with.held_talks"),
                   ("PERSON:antony blinken", "PERSON:wang yi", "met_with.meeting_between")}
    assert not any(r.directed for r in result.relations if r.relation_type == "met_with")


@pytest.mark.parametrize("text", [
    "Donald Trump will meet Vladimir Putin in Alaska next week.",
    "Emmanuel Macron did not meet Volodymyr Zelenskyy during the summit.",
    "Benjamin Netanyahu refused to meet Antonio Guterres.",
    "Donald Trump is expected to meet Vladimir Putin in Alaska.",
    "The Kremlin denied that Vladimir Putin met Donald Trump.",
    "A planned meeting between Xi Jinping and Donald Trump was cancelled.",
])
def test_met_with_suppressed_for_planned_negated_or_denied(run, text):
    result = run(text)
    assert "met_with" not in relation_types(result)
    assert "mentioned_with" in relation_types(result)  # still visible as weak, labelled co-occurrence


def test_criticized_active_passive_and_possessor(run):
    result = run("NATO Secretary General Mark Rutte condemned Russia for the attack.",
                 "Russia was condemned by the United Nations and the European Union.",
                 "President Joe Biden criticised Vladimir Putin's invasion of Ukraine.")
    assert {("PERSON:mark rutte", "criticized", "LOCATION:russia"),
            ("ORG:united nations", "criticized", "LOCATION:russia"),
            ("ORG:european union", "criticized", "LOCATION:russia"),
            ("PERSON:joe biden", "criticized", "PERSON:vladimir putin")} <= semantic(result)
    assert all(r.directed for r in result.relations if r.relation_type == "criticized")


@pytest.mark.parametrize("text", [
    "Germany did not condemn Israel after the strike.",
    "Russia was not condemned by China at the meeting.",
    "Germany could criticize Israel at the meeting.",
])
def test_criticized_suppressed(run, text):
    assert "criticized" not in relation_types(run(text))


def test_criticized_requires_entity_object(run):
    # The object is "the sanctions"; the EU inside its relative clause is not the criticized party.
    result = run("Sergey Lavrov denounced the sanctions imposed by the EU.")
    assert "criticized" not in relation_types(result)


def test_affiliated_with_patterns(run):
    result = run("NATO Secretary General Mark Rutte condemned Russia.",
                 "Kaja Kallas, a member of the European Commission, criticised Hungary.",
                 "Mark Rutte is the secretary general of NATO.",
                 "Iranian Foreign Minister Abbas Araghchi discussed the nuclear programme "
                 "with IAEA chief Rafael Grossi.")
    affiliated = {(r.source_key, r.target_key, r.rule_id) for r in result.relations
                  if r.relation_type == "affiliated_with"}
    assert affiliated == {("PERSON:mark rutte", "ORG:nato", "affiliated_with.title"),
                          ("PERSON:kaja kallas", "ORG:european commission", "affiliated_with.appositive"),
                          ("PERSON:rafael grossi", "ORG:international atomic energy agency", "affiliated_with.title")}
    # The copular sentence repeats an edge already found, so per-item deduplication keeps the first evidence.
    assert next(r for r in result.relations if r.target_key == "ORG:nato").segment_id == "s0"


def test_affiliated_with_suppressed_for_former_roles_and_places(run):
    result = run("Jens Stoltenberg, the former head of NATO, criticized Viktor Orban.",
                 "Iran's foreign minister, Abbas Araghchi, met Rafael Grossi in Vienna.")
    assert "affiliated_with" not in relation_types(result)
    assert ("PERSON:jens stoltenberg", "criticized", "PERSON:viktor orban") in semantic(result)


def test_discussed_topic_verb_and_noun(run):
    result = run("The UN and the U.S. discussed sanctions on North Korea.",
                 "Iran and the United States held talks in Oman on the nuclear programme.")
    assert {("ORG:united nations", "discussed_topic", "TOPIC:sanctions"),
            ("LOCATION:united states", "discussed_topic", "TOPIC:sanctions"),
            ("LOCATION:iran", "discussed_topic", "TOPIC:nuclear programme"),
            ("LOCATION:iran", "met_with", "LOCATION:united states")} <= semantic(result)


def test_discussed_topic_suppressed_when_planned(run):
    result = run("Narendra Modi will discuss trade with Joe Biden in Washington.")
    assert not semantic(result)


def test_cooccurrence_is_sentence_local_and_never_crosses_segments(run):
    result = run("Joe Biden spoke in Berlin. Xi Jinping listened in Beijing.", "Vladimir Putin stayed in Moscow.")
    pairs = {(r.source_key, r.target_key) for r in result.relations}
    assert pairs == {("LOCATION:berlin", "PERSON:joe biden"), ("LOCATION:beijing", "PERSON:xi jinping"),
                     ("LOCATION:moscow", "PERSON:vladimir putin")}
    assert {r.rule_id for r in result.relations} == {"mentioned_with.sentence"}


def test_cooccurrence_skips_semantic_pairs_topic_pairs_and_crowded_sentences(run):
    result = run("Antony Blinken met Benjamin Netanyahu to talk about the ceasefire and sanctions. "
                 "Benjamin Netanyahu and Antony Blinken smiled.",
                 "Officials from the United States, the United Kingdom, France, Germany, Italy, Japan, Canada, "
                 "the European Union and Russia met in Geneva.")
    weak = {(r.source_key, r.target_key) for r in result.relations if r.relation_type == "mentioned_with"}
    assert ("PERSON:antony blinken", "PERSON:benjamin netanyahu") not in weak
    assert ("TOPIC:ceasefire", "TOPIC:sanctions") not in weak
    assert not any(r.segment_id == "s1" for r in result.relations)
    assert any("Co-occurrence skipped" in w for w in result.warnings)


def test_alias_collapse_never_creates_self_edges(run):
    result = run("Joe Biden criticized Biden.")
    assert not result.relations


def test_long_sentence_evidence_is_bounded_and_contains_both_entities(run):
    filler = ", ".join(["after a long day of difficult and detailed negotiations about regional security"] * 8)
    text = f"Antony Blinken met Benjamin Netanyahu in Jerusalem {filler}."
    result = run(text)
    relation = next(r for r in result.relations if r.relation_type == "met_with")
    assert len(relation.evidence_text) <= 500
    assert "Antony Blinken" in relation.evidence_text and "Benjamin Netanyahu" in relation.evidence_text


def test_title_rule_does_not_skip_over_another_entity(run):
    result = run("NATO ally Turkey's President Recep Tayyip Erdogan criticized Sweden.")
    assert "affiliated_with" not in relation_types(result)


# Review regressions (all sentences synthetic)

@pytest.mark.parametrize("text", [
    "A meeting between Putin and Zelensky has not taken place, Peskov said.",
    "The meeting between Erdogan and Assad never happened.",
    "No meeting between Netanyahu and Abbas took place last year.",
    "Egypt called for talks between Israel and Hamas.",
    "Talks between Iran and the US scheduled for Sunday in Muscat were postponed.",
    "Peskov denied that talks between Putin and Zelensky took place.",
])
def test_meeting_nouns_that_did_not_happen_are_suppressed(run, text):
    assert "met_with" not in relation_types(run(text))


def test_meeting_noun_that_happened_still_counts(run):
    assert ("PERSON:assad", "met_with", "PERSON:recep tayyip erdoğan") in semantic(
        run("The meeting between Erdogan and Assad took place in Ankara."))


@pytest.mark.parametrize("text", [
    "Has Donald Trump ever condemned Vladimir Putin?",
    "Did Biden really meet Xi in Bali?",
    "Why did Netanyahu meet Putin in Moscow?",
    "Biden needs to condemn Netanyahu right now.",
    "Biden refrained from criticising Netanyahu, while Macron avoided criticizing Israel.",
    "Biden is going to meet Xi in San Francisco.",
    "Biden has to meet Xi before the summit.",
    "Trump to meet Xi in Seoul",
    "Biden pressed Netanyahu to meet Abbas in Ramallah.",
])
def test_questions_obligations_and_avoided_or_pressed_events_are_suppressed(run, text):
    result = run(text)
    assert not semantic(result)
    assert relation_types(result) <= {"mentioned_with"}


def test_object_control_takes_the_object_as_the_actor(run):
    # "persuade" suppresses, so the controller is checked on a neutral verb ("helped ... to").
    result = run("France helped Qatar to condemn Russia.")
    assert ("LOCATION:qatar", "criticized", "LOCATION:russia") in semantic(result)
    assert ("LOCATION:france", "criticized", "LOCATION:russia") not in semantic(result)


def test_kim_jong_un_does_not_make_un_ambiguous(run):
    result = run("North Korean leader Kim Jong Un oversaw a missile test as the UN Security Council met, "
                 "and the UN condemned Pyongyang.")
    assert ("ORG:united nations", "criticized", "LOCATION:pyongyang") in semantic(result)
    assert not any("Ambiguous" in w for w in result.warnings)


def test_family_name_first_names_resolve_their_short_form(run):
    result = run("Chinese Foreign Minister Wang Yi met Lavrov in Beijing, and Wang later criticised the US.")
    assert ("PERSON:wang yi", "criticized", "LOCATION:united states") in semantic(result)
    assert "ORG:wang" not in keys(result)


@pytest.mark.parametrize(("text", "expected"), [
    ("Trump met Putin in Alaska on Friday.", ("PERSON:donald trump", "met_with", "PERSON:vladimir putin")),
    ("Zelenskyy met Macron in Paris.", ("PERSON:emmanuel macron", "met_with", "PERSON:volodymyr zelenskyy")),
    ("Modi criticised Pakistan after the attack in Pahalgam.",
     ("PERSON:narendra modi", "criticized", "LOCATION:pakistan")),
    ("Trump slammed Zelensky in a post on Truth Social.",
     ("PERSON:donald trump", "criticized", "PERSON:volodymyr zelenskyy")),
])
def test_reviewed_single_word_names_missed_by_ner_are_recovered(run, text, expected):
    assert expected in semantic(run(text))


def test_recovered_single_word_alias_never_splits_another_name(run):
    result = run("Melania Trump met Brigitte Macron in Paris.")
    assert "PERSON:donald trump" not in keys(result) and "PERSON:emmanuel macron" not in keys(result)
    assert "PERSON:donald trump jr." in keys(run("Donald Trump Jr. criticized Zelensky."))
    assert keys(run("Follow #Iran and #Trump for live updates from the talks.")) == set()  # hashtags are tags


@pytest.mark.parametrize(("text", "expected"), [
    ("Netanyahu met Trump at the White House on Monday to discuss Iran's nuclear programme.",
     {("PERSON:benjamin netanyahu", "discussed_topic", "TOPIC:nuclear programme"),
      ("PERSON:donald trump", "discussed_topic", "TOPIC:nuclear programme")}),
    ("The UN Security Council met to discuss the ceasefire in Gaza.",
     {("ORG:united nations security council", "discussed_topic", "TOPIC:ceasefire")}),
])
def test_purpose_clause_after_a_meeting_yields_discussed_topic(run, text, expected):
    assert expected <= semantic(run(text))


def test_comma_separated_actor_lists_keep_every_member(run):
    result = run("The US, the UK, France, Germany, Italy, Japan and Canada condemned Russia's invasion of Ukraine.")
    critics = {s for s, relation, t in semantic(result) if relation == "criticized" and t == "LOCATION:russia"}
    assert critics == {f"LOCATION:{name}" for name in ("united states", "united kingdom", "france", "germany",
                                                       "italy", "japan", "canada")}


def test_title_rule_skips_former_roles_and_accepts_hyphenated_titles(run):
    assert "affiliated_with" not in relation_types(run("Former NATO chief Jens Stoltenberg criticised Russia."))
    assert ("PERSON:antónio guterres", "affiliated_with", "ORG:united nations") in semantic(
        run("UN Secretary-General António Guterres met Iranian President Masoud Pezeshkian in New York."))
    assert ("PERSON:naim qassem", "affiliated_with", "ORG:hezbollah") in semantic(
        run("Hezbollah's Secretary-General Naim Qassem condemned the Israeli strikes, Al Jazeera reported."))


@pytest.mark.parametrize(("text", "junk"), [
    ("Iran's Supreme Leader Ayatollah Ali Khamenei condemned Israel.", "ORG:supreme"),
    ("Trade Minister Piyush Goyal met EU officials as the trade union strike continued.", "ORG:trade"),
    ("u/throwaway_8812 Zelensky met Biden last week, OP is wrong", "ORG:op"),
    ("TIL the US and America and the USA are the same country", "ORG:til"),
    ("EDIT: IMO the EU is useless. NSFW footage from Gaza, AMA.", "ORG:ama"),
    ("Zelenskyy wrote on X, formerly Twitter, that Russia had launched 400 drones.", "PERSON:twitter"),
    ("Donald Trump Jr. criticized Zelensky on X.", "PERSON:x."),
])
def test_title_words_jargon_and_platforms_are_not_entities(run, text, junk):
    result = run(text)
    assert junk not in keys(result)
    assert not any(junk in (r.source_key, r.target_key) for r in result.relations)


def test_person_reading_never_turns_a_place_into_a_person(run):
    result = run("Jordan said the attack was unacceptable. Biden met King Abdullah in Jordan.")
    assert "LOCATION:jordan" in keys(result) and "PERSON:jordan" not in keys(result)


def test_shouted_pronoun_us_is_not_the_united_states(run):
    result = run("SANCTIONS WONT STOP IRAN. LET US BE CLEAR, THE UN HAS FAILED GAZA.")
    assert "LOCATION:united states" not in keys(result) and "ORG:united nations" in keys(result)
    assert "LOCATION:united states" in keys(run("THE US HAS FAILED GAZA."))


@pytest.mark.parametrize(("text", "expected"), [
    ("China condemned US sanctions on Huawei.", ("LOCATION:china", "criticized", "LOCATION:united states")),
    ("The Taliban condemned the US drone strike in Kabul.", ("ORG:taliban", "criticized", "LOCATION:united states")),
    ("Iran condemned the attack by Israel on its consulate in Damascus.",
     ("LOCATION:iran", "criticized", "LOCATION:israel")),
])
def test_criticized_reaches_the_actor_behind_the_criticized_action(run, text, expected):
    assert expected in semantic(run(text))


def test_criticized_ignores_place_and_target_modifiers(run):
    assert "criticized" not in relation_types(run("Erdoğan condemned the Gaza attack."))
    assert "criticized" not in relation_types(run("China condemned the Iran sanctions."))


def test_line_breaks_are_sentence_boundaries(run):
    result = run("Biden is weak\nPutin is laughing\nXi is watching\nIran is next")
    assert result.relations == []


def test_reviewed_alias_table_lists_aliases_and_handles_with_and_without_at(extractor):
    musk = dict(extractor.reviewed["PERSON:elon musk"])
    assert musk == {"musk": "alias", "@elonmusk": "handle", "elonmusk": "handle"}
    assert dict(extractor.reviewed["ORG:united nations"])["un"] == "abbreviation"
    assert "united nations" not in dict(extractor.reviewed["ORG:united nations"])

# AI-industry vocabulary (decision D23). All sentences synthetic.

@pytest.mark.parametrize(("text", "topic"), [
    ("Commenters said the new LLMs are mostly marketing.", "TOPIC:ai models"),
    ("The company delayed its IPO again.", "TOPIC:funding and deals"),
    ("Demand for GPUs keeps rising.", "TOPIC:chips and compute"),
    ("The paper is about AI safety and existential risk.", "TOPIC:ai safety"),
    ("Coding agents now write most of the patches.", "TOPIC:ai agents"),
    ("The lawsuit claims copyright infringement.", "TOPIC:copyright"),
    ("The startup announced layoffs on Friday.", "TOPIC:jobs and layoffs"),
])
def test_ai_topics_match_whole_phrases(run, text, topic):
    assert topic in keys(run(text))


@pytest.mark.parametrize("text", [
    "Iran uses the strait as a bargaining chip, a familiar business model.",  # not chips, not ai models
    "FBI agents searched the office after the Taliban takeover.",  # not ai agents, not funding and deals
    "Clinton banned trade and investment with the country.",  # "investment" alone is not a deal topic
    "This is Hacker News, where a hack is usually a clever workaround.",  # bare "hack" is not cybersecurity
])
def test_ai_topics_reject_other_senses(run, text):
    assert not {k for k in keys(run(text)) if k.startswith("TOPIC:") and k != "TOPIC:trade"}


def test_tech_acronyms_are_not_entities_and_products_are_orgs(run):
    result = run("The AI startup hired a new CEO to run its LLM and API business.",
                 "I asked Claude and Meta's chatbot, then thanked @sama for it.")
    assert not {k for k in keys(result) if k.split(":")[1] in {"ai", "ceo", "llm", "api"}}
    assert {"ORG:claude", "ORG:meta", "PERSON:sam altman"} <= keys(result)
    assert "PERSON:claude" not in keys(result) and "PERSON:meta" not in keys(result)


def test_ner_span_that_is_a_topic_phrase_becomes_the_topic(run):
    result = run("Since then the EU has passed the AI Act.", "So why is OpenAI not charged with Cybercrimes.")
    assert {"TOPIC:tech regulation", "TOPIC:cybersecurity"} <= keys(result)
    assert not {"ORG:ai act", "ORG:cybercrimes"} & keys(result)


def test_ai_company_aliases_collapse_to_one_node(run):
    result = run("HF is an American company.", "Hugging Face sold to NVIDIA.", "Nvidia paid for Huggingface.")
    assert {"ORG:hugging face", "ORG:nvidia"} <= keys(result)
    assert not {"ORG:hf", "ORG:huggingface", "LOCATION:nvidia"} & keys(result)


# acquired

def test_acquired_verb_passive_noun_and_announced_deal(run):
    result = run("Microsoft acquired GitHub in 2018.",
                 "Figma was acquired by Adobe for $20 billion.",
                 "Microsoft's acquisition of Activision closed after a long review.",
                 "Nvidia agrees to acquire Hugging Face for $13B")
    acquired = {(r.source_key, r.target_key, r.rule_id) for r in result.relations if r.relation_type == "acquired"}
    assert ("ORG:microsoft", "ORG:github", "acquired.verb") in acquired
    assert ("ORG:adobe", "ORG:figma", "acquired.passive") in acquired
    assert ("ORG:nvidia", "ORG:hugging face", "acquired.verb") in acquired  # "agrees to" is an announced deal
    assert any(rule == "acquired.noun" and source == "ORG:microsoft" for source, _, rule in acquired)
    assert all(r.directed for r in result.relations if r.relation_type == "acquired")


@pytest.mark.parametrize("text", [
    "Nvidia is in talks to acquire Hugging Face.",
    "Nvidia plans to buy Hugging Face.",
    "Nvidia did not acquire Hugging Face.",
    "Nvidia could acquire Hugging Face next year.",
    "Nvidia refused to agree to acquire Hugging Face.",
    "Nvidia to acquire Hugging Face",  # headline infinitive: announced intent, not a done deal
    "Did Nvidia acquire Hugging Face?",
    "Nvidia's attempted acquisition of Arm collapsed.",
])
def test_acquired_suppressed_for_talks_plans_negation_and_questions(run, text):
    assert "acquired" not in relation_types(run(text))


def test_acquired_direction_and_types(run):
    assert ("ORG:hugging face", "acquired", "ORG:nvidia") not in semantic(run("Nvidia bought Hugging Face."))
    # Countries are not acquired; buying a company's products is not an acquisition of the company.
    assert "acquired" not in relation_types(run("Iran acquired Russia's jets. Meta buys Nvidia chips by the thousands."))


# invested_in

@pytest.mark.parametrize(("text", "expected", "rule"), [
    ("Microsoft invested $10 billion in OpenAI.", ("ORG:microsoft", "ORG:openai"), "invested_in.verb"),
    ("SoftBank took a stake in Anthropic.", ("ORG:softbank", "ORG:anthropic"), "invested_in.noun"),
    ("Amazon's investment in Anthropic paid off.", ("ORG:amazon", "ORG:anthropic"), "invested_in.noun"),
    ("Mistral raised $600 million from Nvidia.", ("ORG:nvidia", "ORG:mistral ai"), "invested_in.raised_from"),
    ("OpenAI raised $40 billion in a round led by SoftBank.", ("ORG:softbank", "ORG:openai"),
     "invested_in.round_led_by"),
    ("Microsoft is OpenAI's largest investor.", ("ORG:microsoft", "ORG:openai"), "invested_in.copular"),
    ("Salesforce, an investor in Hugging Face, declined to comment.", ("ORG:salesforce", "ORG:hugging face"),
     "invested_in.appositive"),
])
def test_invested_in_variants_point_from_investor_to_company(run, text, expected, rule):
    found = {(r.source_key, r.target_key, r.rule_id, r.directed) for r in run(text).relations
             if r.relation_type == "invested_in"}
    assert found == {(*expected, rule, True)}


@pytest.mark.parametrize("text", [
    "Hugging Face turned down a $500 million investment from Nvidia.",
    "Microsoft did not invest in Anthropic.",
    "Apple may invest in OpenAI.",
    "Google refused to invest in Mistral.",
    "Anthropic raised concerns from Microsoft about the contract.",  # no money object: not a funding round
])
def test_invested_in_suppressed(run, text):
    assert "invested_in" not in relation_types(run(text))


# partnered_with

def test_partnered_with_variants_are_symmetric_and_org_only(run):
    result = run("Apple partnered with OpenAI to bring ChatGPT to the iPhone.",
                 "Samsung and Google teamed up on a new headset.",
                 "Nvidia signed a deal with Oracle to supply GPUs.",
                 "Microsoft's partnership with OpenAI is under strain.",
                 "Iran signed an agreement with the United States.")
    partnered = {(r.source_key, r.target_key) for r in result.relations if r.relation_type == "partnered_with"}
    assert {("ORG:apple", "ORG:openai"), ("ORG:google", "ORG:samsung"), ("ORG:nvidia", "ORG:oracle"),
            ("ORG:microsoft", "ORG:openai")} <= partnered
    assert not any(r.directed for r in result.relations if r.relation_type == "partnered_with")
    assert not any("LOCATION:" in pair[0] or "LOCATION:" in pair[1] for pair in partnered)


@pytest.mark.parametrize("text", [
    "Wikimedia does not have a deal with OpenAI.",
    "Apple is in talks to partner with Google.",
    "Apple wants to sign a deal with Google.",
])
def test_partnered_with_suppressed(run, text):
    assert "partnered_with" not in relation_types(run(text))


# affiliated_with (AI-industry shapes)

@pytest.mark.parametrize(("text", "rule"), [
    ("OpenAI CEO Sam Altman met Satya Nadella.", "affiliated_with.title"),
    ("Sam Altman, OpenAI's chief executive, criticized Elon Musk.", "affiliated_with.appositive"),
    ("Anthropic's Dario Amodei warned about AI safety.", "affiliated_with.possessive"),
    ("Nvidia founder Jensen Huang spoke at the conference.", "affiliated_with.title"),
    ("OpenAI co-founder Ilya Sutskever left the company.", "affiliated_with.title"),
    ("Demis Hassabis co-founded DeepMind in 2010.", "affiliated_with.founded"),
    ("Google DeepMind was founded by Demis Hassabis.", "affiliated_with.founded"),
    ("Altman, who runs OpenAI, testified before Congress.", "affiliated_with.runs"),
    ("Jane Smith works at Anthropic as a researcher.", "affiliated_with.works_for"),
    ("Muhanad Seloom of the Doha Institute of Graduate Studies spoke on Tuesday.", "affiliated_with.of"),
])
def test_affiliated_with_ai_industry_shapes(run, text, rule):
    result = run(text)
    assert rule in {r.rule_id for r in result.relations if r.relation_type == "affiliated_with"}
    assert all(r.target_key.startswith("ORG:") for r in result.relations if r.relation_type == "affiliated_with")


@pytest.mark.parametrize("text", [
    "Former OpenAI CTO Mira Murati spoke at the event.",
    "Russia's deputy UN envoy, Dmitry Polyanskiy, spoke to the council.",  # represents Russia at the UN
    "OpenAI rival Anthropic released a model.",  # "rival" is not a role
])
def test_affiliated_with_rejects_former_roles_envoys_and_rivals(run, text):
    assert "affiliated_with" not in relation_types(run(text))


# discussed_topic

def test_discussed_topic_fires_on_announcements_and_speech(run):
    result = run("Sundar Pichai announced layoffs at Google.",
                 "Satya Nadella spoke about AI regulation at the summit.",
                 "Google announces Gemini 4 AI model, but you can't use it yet")
    assert {("PERSON:sundar pichai", "discussed_topic", "TOPIC:jobs and layoffs"),
            ("PERSON:satya nadella", "discussed_topic", "TOPIC:tech regulation")} <= semantic(result)


def test_discussed_topic_ignores_adjuncts_outside_the_object(run):
    result = run("The United States discussed new sanctions on Iran during the war.")
    assert ("LOCATION:united states", "discussed_topic", "TOPIC:sanctions") in semantic(result)
    assert ("LOCATION:united states", "discussed_topic", "TOPIC:military conflict") not in semantic(result)


def test_announced_sanctions_are_sanctioned_not_also_discussed(run):
    # The precision review found every "announced sanctions" sentence duplicated as discussed_topic -> sanctions.
    result = run("The United States announced new sanctions on Iran during the war.")
    assert semantic(result) == {("LOCATION:united states", "sanctioned", "LOCATION:iran")}
    # A product announcement still names its topic; a reason ("over ...") never does.
    assert ("ORG:google", "discussed_topic", "TOPIC:ai models") in semantic(
        run("Google announces Gemini 4 Argon AI model, but you can't use it yet"))
    over = semantic(run("The US announced new sanctions on Iran over its nuclear programme."))
    assert ("LOCATION:united states", "discussed_topic", "TOPIC:nuclear programme") not in over


def test_discussed_topic_skips_settings_and_reasons(run):
    result = run("Abbas Araghchi spoke to reporters on the sidelines of the nuclear talks in Geneva.",
                 "The IRGC announced the seizure of a tanker over sanctions violations.")
    assert not {r for r in semantic(result) if r[1] == "discussed_topic"}


def test_discussed_topic_from_the_owner_of_a_non_entity_subject(run):
    result = run("Anthropic’s IPO pitch includes a warning about human extinction")
    assert ("ORG:anthropic", "discussed_topic", "TOPIC:ai safety") in semantic(result)


# sanctioned

def test_sanctioned_variants(run):
    result = run("The EU imposed sanctions on Russia.",
                 "UN sanctions were reimposed against Iran in September.",
                 "The US issued new Syria sanctions on Tuesday.",
                 "Cuba has been under US sanctions since 1962.",
                 "The UK's sanctions regime against Belarus largely mirrors those of the EU.",
                 "The US sanctioned several firms, including RPT Technology Ltd and Hezbollah.")
    assert {("ORG:european union", "sanctioned", "LOCATION:russia"),
            ("ORG:united nations", "sanctioned", "LOCATION:iran"),
            ("LOCATION:united states", "sanctioned", "LOCATION:syria"),
            ("LOCATION:united states", "sanctioned", "LOCATION:cuba"),
            ("LOCATION:united kingdom", "sanctioned", "LOCATION:belarus"),
            ("LOCATION:united states", "sanctioned", "ORG:hezbollah")} <= semantic(result)
    assert all(r.directed for r in result.relations if r.relation_type == "sanctioned")


@pytest.mark.parametrize("text", [
    "The EU lifted its sanctions on Syria.",
    "The US threatened sanctions on China.",
    "Washington called for sanctions on Moscow.",
    "The US issued a sanctions waiver for Iran.",
    "The EU did not impose sanctions on Israel.",
    "Brussels may impose sanctions on Hungary.",
])
def test_sanctioned_suppressed_for_lifted_threatened_demanded_or_waived(run, text):
    assert "sanctioned" not in relation_types(run(text))


# attacked

def test_attacked_variants(run):
    result = run("Hamas attacked Israel in October.",
                 "Israel launched airstrikes on Lebanon.",
                 "Russia's invasion of Ukraine began in 2022.",
                 "Hezbollah's attacks on Israel continued.",
                 "Tehran launched retaliatory attacks on US bases in Bahrain.")
    assert {("ORG:hamas", "attacked", "LOCATION:israel"), ("LOCATION:israel", "attacked", "LOCATION:lebanon"),
            ("LOCATION:russia", "attacked", "LOCATION:ukraine"), ("ORG:hezbollah", "attacked", "LOCATION:israel"),
            ("LOCATION:tehran", "attacked", "LOCATION:united states")} <= semantic(result)
    # The bases are the US's; Bahrain is only where they are.
    assert ("LOCATION:tehran", "attacked", "LOCATION:bahrain") not in semantic(result)
    assert ("LOCATION:israel", "attacked", "ORG:hamas") not in semantic(result)


@pytest.mark.parametrize("text", [
    "Trump attacked Biden in a speech.",  # people attack each other verbally
    "Iran warned that any US strike on Iran would trigger retaliation.",
    "The US deterred Iran from attacking Israel.",
    "Israel did not strike Iran on Friday.",
    "Iran threatened to attack Israel.",
    "OpenAI agents tried to hack Wikipedia tools.",
])
def test_attacked_suppressed_for_people_hypotheticals_and_threats(run, text):
    assert "attacked" not in relation_types(run(text))


# quoted_by

def test_quoted_by_needs_a_news_outlet(run):
    result = run("Elon Musk told Reuters that xAI would raise money.",
                 "Tim Hawkins told Al Jazeera English’s Tom McRae that the strait was open.",
                 "Abbas Araghchi told the UN Security Council that the snapback was void.")
    assert {("PERSON:elon musk", "quoted_by", "ORG:reuters"),
            ("PERSON:tim hawkins", "quoted_by", "ORG:al jazeera")} <= semantic(result)
    assert not any(r.relation_type == "quoted_by" and r.target_key == "ORG:united nations security council"
                   for r in result.relations)


def test_quoted_by_suppressed_when_refused(run):
    assert "quoted_by" not in relation_types(run("Elon Musk declined to tell Reuters about xAI."))


def test_every_relation_type_is_defined():
    assert DIRECTED < set(RELATION_DEFINITIONS)
    assert {"met_with", "partnered_with", "mentioned_with"} == set(RELATION_DEFINITIONS) - DIRECTED


# Calibration corpus regressions (quoted from the 7 October 2026 crawl) and the precision review's probe sentences

def test_bare_foreign_ministry_takes_the_country_written_next_to_it_or_is_dropped(run):
    result = run("Iran’s Ministry of Foreign Affairs strongly condemned the “aggressive attacks and gross violation” "
                 "of the MoU by the US.",
                 "Qatar’s Foreign Ministry also strongly condemned Iranian attacks on Bahrain and Kuwait.",
                 "“The talks will continue,” Esmaeil Baghaei, the Iranian Foreign Ministry spokesman, said on Monday.",
                 "Ministry of Foreign Affairs spokesman Lin Jian told reporters on Monday.")
    assert {k for k in keys(result) if "ministry" in k} == {"ORG:foreign ministry (iran)", "ORG:foreign ministry (qatar)"}
    assert {("ORG:foreign ministry (iran)", "criticized", "LOCATION:united states"),
            ("ORG:foreign ministry (qatar)", "criticized", "LOCATION:iran"),
            ("PERSON:esmaeil baghaei", "affiliated_with", "ORG:foreign ministry (iran)")} <= semantic(result)
    assert not any(r[0] == "PERSON:lin jian" for r in semantic(result))  # whose ministry is not written


def test_nationality_adjective_and_place_compound_name_the_attacker(run):
    result = run("Qatar’s Foreign Ministry also strongly condemned Iranian attacks on Bahrain and Kuwait.",
                 "Continued Israeli attacks on Lebanon all represent major violations.",
                 "Oil prices rose because of the United States-Israel war on Iran.")
    assert {("LOCATION:iran", "attacked", "LOCATION:bahrain"), ("LOCATION:iran", "attacked", "LOCATION:kuwait"),
            ("LOCATION:israel", "attacked", "LOCATION:lebanon"),
            ("LOCATION:israel", "attacked", "LOCATION:iran")} <= semantic(result)
    assert "LOCATION:iran" in keys(run("Iranian state media said the vessel was seized."))


def test_claims_and_rumours_are_not_asserted_events(run):
    assert not semantic(run("Iran claimed Israel attacked Iran.", "Rumours that Nvidia bought Hugging Face are false."))


def test_quoted_by_needs_a_news_outlet_and_a_statement(run):
    assert not semantic(run("Araghchi told the IAEA that Iran would continue to cooperate.",
                            "Pete Hegseth told the National Security Agency to stand down."))
    result = run("An unnamed Pentagon spokesperson confirmed to the Reuters news agency the loss of the submersible.",
                 "Sergey Lavrov spoke to Agence France-Presse on Monday.")
    assert {("ORG:pentagon", "quoted_by", "ORG:reuters"),
            ("PERSON:sergey lavrov", "quoted_by", "ORG:agence france-presse")} <= semantic(result)


def test_attacked_needs_a_place_and_is_not_criticism(run):
    result = run("China attacked the United States over its new tariffs.",
                 "OpenAI attacked Anthropic in a blog post on Tuesday.",
                 "The IRGC attacked a US vessel near the Strait of Hormuz.")
    assert {r for r in semantic(result) if r[1] == "attacked"} == {
        ("ORG:islamic revolutionary guard corps", "attacked", "LOCATION:united states")}


def test_hosting_is_a_meeting_only_with_a_person(run):
    result = run("Qatar hosts Hamas.", "Bahrain, which hosts CENTCOM, condemned the attack.",
                 "Emmanuel Macron hosted Olaf Scholz in Paris on Friday.")
    assert {r for r in semantic(result) if r[1] == "met_with"} == {
        ("PERSON:emmanuel macron", "met_with", "PERSON:olaf scholz")}


def test_affiliation_needs_a_current_role_held_by_a_person(run):
    result = run("John Sculley was the CEO of Apple.", "Jan Leike worked at OpenAI until 2024.",
                 "OpenAI leads Google in the race to build AI agents.",
                 "Ex-Google engineer Blake Lemoine spoke to reporters.", "Sam Altman is the CEO of OpenAI.")
    assert {r for r in semantic(result) if r[1] == "affiliated_with"} == {
        ("PERSON:sam altman", "affiliated_with", "ORG:openai")}
    assert not any("ex-" in k for k in keys(result))


def test_released_needs_a_product_noun(run):
    result = run("Google announces Gemini 4 Argon AI model, but you can't use it yet",
                 "The Houthis released the Galaxy Leader crew.")
    assert {r for r in semantic(result) if r[1] == "released"} == {("ORG:google", "released", "ORG:gemini")}
    assert "released" in RELATION_DEFINITIONS and "released" in DIRECTED


def test_item_defined_abbreviation_and_particle_surname_resolve(run):
    result = run("Iran's Supreme National Security Council (SNSC) met on Sunday.",
                 "The SNSC said the strikes violated the agreement.",
                 "Alfred-Maurice de Zayas wrote the report for the council.",
                 "The sanctions are illegal, according to de Zayas.")
    council = [(m.name, m.resolution) for m in result.mentions if m.key == "ORG:supreme national security council"]
    assert ("SNSC", "abbreviation") in council and len(council) >= 2
    assert [m.resolution for m in result.mentions if m.key == "PERSON:alfred-maurice de zayas"] == ["exact", "surname"]
    assert not {k for k in keys(result) if k in ("ORG:snsc", "PERSON:de zayas")}


def test_one_label_per_name_within_an_item(run):
    assert keys(run("At the Muscat meeting, envoys agreed on a timetable.",
                    "The embassy in Muscat confirmed the timetable.")) == {"LOCATION:muscat"}


def test_photo_credits_headings_and_common_words_are_not_entities(run):
    result = run("Smoke rises over the port after the strike [Murtaja Lateef/AFP]",
                 "Tankers near the coast [US Central Command/Handout via Reuters]",
                 "Major MOU Violations by the US:", "The MoU was signed in June and the MoU covers shipping.",
                 "Strikes on the port resumed overnight.", "The strikes killed two people, the law was clear.")
    assert keys(result) == {"LOCATION:united states"}


def test_reviewed_handle_in_an_embedded_post_byline_resolves_to_one_person(run):
    result = run("— محمدباقر قالیباف | MB Ghalibaf (@mb_ghalibaf) September 8, 2026",
                 "Mohammad Bagher Ghalibaf said the talks failed.")
    assert keys(result) == {"PERSON:mohammad bagher ghalibaf"}
    assert {m.resolution for m in result.mentions} == {"alias", "handle", "exact"}


def test_list_and_fragment_spans_are_rejected(extractor):
    doc = extractor.nlp("Lego (Denmark makes toys. ML/AI is hot. Vienna Shipping Co and others.")
    doc.ents = [spacy.tokens.Span(doc, 0, 3, label="ORG"), spacy.tokens.Span(doc, 6, 9, label="ORG"),
                spacy.tokens.Span(doc, 12, 16, label="ORG")]
    assert [m.span.text for m in extractor._candidates(doc)] == ["Vienna Shipping Co"]


def test_headline_comma_list_is_not_an_affiliation(run):
    """Real headline (README edge #860): the comma means "and", so no organisation is affiliated with the other."""
    result = run("Nvidia, Supermicro employees charged over export of AI servers to China")
    assert not any(rel == "affiliated_with" for _, rel, _ in semantic(result))
    person = run("Esmaeil Baghaei, the Iranian Foreign Ministry spokesman, wrote on X.")
    assert any(rel == "affiliated_with" and src == "PERSON:esmaeil baghaei" for src, rel, _ in semantic(person))


def test_reviewed_alias_fixes_company_type_across_items(run):
    """Real sentences where spaCy typed Cloudflare as ORG in one comment and as a place in another."""
    first = run("I blocked by IP range, and eventually put Cloudflare up.")
    second = run("That puts it in the Cloudflare of AI models category.")
    assert "ORG:cloudflare" in keys(first) and "ORG:cloudflare" in keys(second)
    assert "LOCATION:cloudflare" not in keys(first) | keys(second)
