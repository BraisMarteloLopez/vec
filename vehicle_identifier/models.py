"""Pydantic schemas for the Fase 1 extraction pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Feature(BaseModel):
    """A single discriminating characteristic, curated by a human expert.

    One Feature -> one isolated prompt at runtime.
    """

    id: str
    category: str
    question: str
    context: str
    examples: list[str] = Field(default_factory=list)


class FeatureResult(BaseModel):
    """The model's answer for one characteristic."""

    feature_id: str
    value: str
    reasoning: str
    raw_response: str = ""
    error: str | None = None


class ExtractionState(BaseModel):
    """State threaded through the Fase 1 pipeline."""

    image_path: str
    image_b64: str
    image_mime: str = "image/jpeg"
    features: list[Feature] = Field(default_factory=list)
    results: list[FeatureResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
