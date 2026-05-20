"""Fase 1: recorre el catálogo lanzando una inferencia aislada por característica."""

from __future__ import annotations

import json
import re

from .client import Client
from .models import ExtractionState, FeatureResult
from .prompts import render_feature_prompt

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from an LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # strip ```json ... ``` fences
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(cleaned)
        if match:
            return json.loads(match.group(0))
        raise


def extract(state: ExtractionState, client: Client) -> ExtractionState:
    """Run one isolated prompt per feature, accumulating results in state."""
    for feature in state.features:
        prompt = render_feature_prompt(feature)
        raw = ""
        try:
            raw = client.ask(prompt, image_b64=state.image_b64, mime=state.image_mime)
            data = parse_json(raw)
            state.results.append(
                FeatureResult(
                    feature_id=feature.id,
                    value=str(data.get("value", "")),
                    reasoning=str(data.get("reasoning", "")),
                    raw_response=raw,
                )
            )
        except Exception as exc:  # noqa: BLE001 — errores no fatales por característica
            msg = f"{feature.id}: {type(exc).__name__}: {exc}"
            state.errors.append(msg)
            state.results.append(
                FeatureResult(
                    feature_id=feature.id,
                    value="",
                    reasoning="",
                    raw_response=raw,
                    error=msg,
                )
            )
    return state
