"""Shared data contracts; external content is always treated as data."""

import json
import unicodedata
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

EntityType = Literal["PERSON", "ORG", "LOCATION", "TOPIC"]
SourceType = Literal["news", "discussion", "microblog", "official", "analysis"]


def utcnow() -> datetime:
    return datetime.now(UTC)


def normalize_name(value: str) -> str:
    """Lookup form of a name: NFKC, casefolded, single-spaced; display names are kept separately."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def node_key(entity_type: str, name: str) -> str:
    """Stable, type-aware identity so a PERSON and an ORG with the same name never merge."""
    return f"{entity_type}:{normalize_name(name)}"


def timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("A timestamp must include a timezone")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class Segment(BaseModel):
    id: str
    text: str
    kind: str = "body"
    author: str | None = None
    parent_id: str | None = None
    url: str | None = None
    published_at: datetime | None = None


class ContentItem(BaseModel):
    source_url: str
    source_type: SourceType
    scraped_at: datetime
    title: str | None = None
    body: str = Field(min_length=1)
    author: str | None = None
    published_at: datetime | None = None
    requested_url: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @field_validator("scraped_at", "published_at")
    @classmethod
    def aware_time(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            if value.tzinfo is None:
                raise ValueError("Content timestamps must include a timezone")
            return value.astimezone(UTC)
        return None

    @property
    def content_hash(self) -> str:
        # Deliberately excludes observation time, votes, and other volatile metadata.
        value = {"title": self.title, "body": self.body,
                 "segments": [s.model_dump(mode="json") for s in self.segments]}
        return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class EntityMention(BaseModel):
    name: str
    entity_type: EntityType
    canonical_name: str
    key: str
    segment_id: str
    start: int
    end: int
    resolution: str = "exact"


class Relation(BaseModel):
    source_key: str
    target_key: str
    relation_type: str
    directed: bool = True
    evidence_text: str
    segment_id: str
    start: int
    end: int
    rule_id: str
    quality_tier: Literal["semantic_rule", "cooccurrence"]


class Extraction(BaseModel):
    mentions: list[EntityMention] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FetchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    requested_url: str
    final_url: str
    source_name: str
    depth: int
    fetched_at: datetime = Field(default_factory=utcnow)
    success: bool = False
    status_code: int | None = None
    html: str = ""
    links: list[str] = Field(default_factory=list)
    error: str | None = None

