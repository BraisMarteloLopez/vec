"""Prompt templates and rendering for Fase 1.

Draft prompt: the model should *reason* about one concrete characteristic.
No confidence score is requested.
"""

from __future__ import annotations

from .models import Feature

FEATURE_PROMPT_TEMPLATE = """\
Analiza esta imagen y céntrate EXCLUSIVAMENTE en una característica del vehículo.

Característica: {question}

En qué fijarte: {context}

Ejemplos de lo que se busca identificar:
{examples}

Razona sobre lo que observas y responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"value": "<descripción concreta de lo observado>",
  "reasoning": "<razonamiento: qué ves y por qué; indica si no hay evidencia suficiente>"}}
"""


def render_examples(examples: list[str]) -> str:
    if not examples:
        return "- (sin ejemplos)"
    return "\n".join(f"- {e}" for e in examples)


def render_feature_prompt(feature: Feature) -> str:
    return FEATURE_PROMPT_TEMPLATE.format(
        question=feature.question,
        context=feature.context,
        examples=render_examples(feature.examples),
    )
