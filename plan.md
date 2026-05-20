# Vehicle Identifier — Arquitectura del Sistema

## Visión General

Sistema para identificar un vehículo a partir de una imagen (objetivo inicial:
**helicópteros**). En lugar de preguntar de forma abierta *"¿qué modelo es?"*, el
sistema recorre un **catálogo de características** definido por un experto y, **por
cada característica, lanza una pregunta aislada** al modelo multimodal —una
inferencia independiente por característica—. Cada respuesta se guarda junto con el
razonamiento del modelo.

**No hay clasificación previa del tipo de vehículo.** El flujo va directamente al
catálogo y lo recorre característica a característica.

La **agregación** de las respuestas y la **identificación final** del modelo **están
aún por definir** (ver sección correspondiente).

### Reparto de trabajo

- **Trabajo humano (offline):** definir el **catálogo de características** del vehículo
  (helicóptero). Por cada característica: la **pregunta** concreta, el **contexto** (en
  qué fijarse al mirar la imagen) y, opcionalmente, **ejemplos** de lo que se busca.
- **Trabajo autónomo (runtime):** recorrer el catálogo y, **por cada característica,
  lanzar un único prompt** (pregunta + contexto + ejemplos) al modelo, de forma
  **aislada e independiente**, guardando la respuesta y su razonamiento.

### Hipótesis de diseño

`Gemma-4` detecta rasgos concretos con alta fiabilidad cuando la pregunta es
**estrecha, aislada y acompañada de ejemplos**. Una inferencia por característica
produce una validación más fiable y auditable que una sola pregunta abierta de
identificación, y deja un rastro de evidencia por cada detalle.

### Fases

| Fase | Alcance | Estado |
|---|---|---|
| **1 — Extracción de características** | Catálogo (característica + contexto + ejemplos) y recorrido que lanza **un prompt aislado por característica** y guarda cada respuesta | **Definido — foco actual** |
| **2 — Agregación e identificación** | Consolidar las respuestas y deducir el modelo concreto | **Por definir** |

---

## Flujo (Fase 1)

```
imagen del vehículo
        │
        ▼
recorrer el CATÁLOGO de características
  · por cada característica → 1 prompt aislado (pregunta + contexto + ejemplos)
  · cada respuesta → value + reasoning
        │
        ▼
lista de respuestas (una por característica)
        │
  ───────────────────────  fin de la Fase 1  ───────────────────────
        │
        ▼
agregación + identificación del modelo   →   POR DEFINIR
```

Cada característica se pregunta **una sola vez y por separado**: no hay clasificación
previa, ni re-preguntas, ni dependencia entre características.

---

## Catálogo de Características (lo aporta el humano)

El experto define esta tabla. Cada fila es una característica que el flujo preguntará
de forma aislada. (Extracto de ejemplo para helicóptero; el catálogo real es más amplio.)

| id | Pregunta (característica) | Contexto — en qué fijarse | Ejemplos de lo que se busca |
|---|---|---|---|
| `main_rotor_blade_count` | ¿Cuántas palas tiene el rotor principal y qué forma/anchura tienen? | Cuenta las palas del cubo del rotor principal (eje vertical sobre el fuselaje); ignora el rotor de cola. Fíjate en el ancho de la pala y si la punta es recta, en flecha o doblada. | 3 palas rectas y estrechas · 5 palas anchas con punta en flecha · palas con doblez en la punta (tip curl) |
| `tail_rotor_config` | ¿Qué tipo de rotor de cola tiene? | Mira la base del estabilizador vertical. Distingue rotor convencional expuesto, fenestron (rotor carenado), NOTAR (sin rotor visible) o coaxial (sin rotor de cola). | convencional de 2-4 palas expuesto · fenestron · NOTAR |
| `engine_count_position` | ¿Cuántos motores tiene y dónde están montados? | Localiza los carenados de motor y las tomas/escapes; en helicópteros van sobre el fuselaje, laterales o en la nariz. | 1 sobre el fuselaje · 2 laterales · 2 con tomas sobre la cabina |
| … | … | … | … |

En código, cada fila es un objeto `Feature` (ver schema). El catálogo vive en
`config/features.py`.

---

## Stack Técnico

| Componente | Librería | Motivo |
|---|---|---|
| Orquestación | `langgraph` | Encadenar los pasos del pipeline (y la futura Fase 2) |
| Schema / validación | `pydantic v2` | Estado tipado y parsing de las respuestas JSON |
| LLM client | `openai` | Compatible con el endpoint vLLM local |
| Modelo | `nvidia/Gemma-4-31B-IT-NVFP4` | Multimodal, local, ya operativo |

```bash
pip install langgraph pydantic openai
```

En la Fase 1 el flujo es **lineal** (recorrer el catálogo); LangGraph cobrará más
sentido al añadir la Fase 2.

---

## Estructura de Ficheros

```
vehicle_identifier/
│
├── config/
│   └── features.py        # Catálogo de características (helicópteros)
│
├── models.py              # Pydantic: Feature, FeatureResult, ExtractionState
├── client.py              # Wrapper del cliente vLLM (reutiliza la lógica de _test_.py)
├── prompts.py             # FEATURE_PROMPT_TEMPLATE
├── extractor.py           # Recorre el catálogo: 1 prompt aislado por característica
└── run.py                 # Entry point CLI
│
# Agregación e identificación → ficheros por definir (Fase 2)
```

---

## Schemas Pydantic

```python
# models.py

class Feature(BaseModel):
    id: str                  # "main_rotor_blade_count"
    category: str            # "rotor" | "tail" | "propulsion" | ...
    question: str            # Pregunta concreta sobre ESTA característica
    context: str             # Guía de en qué fijarse al observar la imagen
    examples: list[str]      # Ejemplos concretos de lo que se busca

class FeatureResult(BaseModel):
    feature_id: str
    value: str               # Respuesta concreta sobre la característica
    reasoning: str           # Razonamiento del modelo sobre esta característica
    raw_response: str        # Respuesta completa del LLM (para auditoría)

class ExtractionState(BaseModel):
    image_path: str
    image_b64: str           # cargado una vez, reutilizado en cada llamada
    features: list[Feature]        # catálogo a recorrer
    results: list[FeatureResult]   # una entrada por característica
    errors: list[str]              # errores no fatales acumulados
```

---

## Recorrido del Catálogo (Fase 1)

```python
# extractor.py — núcleo de la Fase 1 (esbozo)

def extract(state: ExtractionState) -> ExtractionState:
    for feature in state.features:
        prompt = FEATURE_PROMPT_TEMPLATE.format(
            question=feature.question,
            context=feature.context,
            examples="\n".join(f"- {e}" for e in feature.examples),
        )
        raw = client.ask(prompt, image_b64=state.image_b64)   # inferencia aislada
        data = parse_json(raw)
        state.results.append(FeatureResult(
            feature_id=feature.id,
            value=data["value"],
            reasoning=data["reasoning"],
            raw_response=raw,
        ))
    return state
```

Cada iteración es **independiente**: una llamada por característica, sin estado
compartido entre preguntas.

---

## Prompt de Característica

```python
# prompts.py  (borrador — pendiente de diseño fino)

FEATURE_PROMPT_TEMPLATE = """
Analiza esta imagen y céntrate EXCLUSIVAMENTE en una característica del vehículo.

Característica: {question}

En qué fijarte: {context}

Ejemplos de lo que se busca identificar:
{examples}

Razona sobre lo que observas y responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"value": "<descripción concreta de lo observado>",
  "reasoning": "<razonamiento: qué ves y por qué; indica si no hay evidencia suficiente>"}}
"""
```

> **Nota.** El prompt es un **borrador pendiente de diseño**: la idea es que el modelo
> **razone** sobre la característica concreta (no que emita una confianza numérica). Se
> rellena con `str.format()`, por lo que las llaves literales del JSON van escapadas como
> `{{ }}` y `{examples}` se renderiza como lista antes de formatear. El forzado de JSON
> conviene apoyarlo en la salida estructurada del endpoint (vLLM `guided_json` /
> `response_format`).

---

## Agregación e Identificación — POR DEFINIR

Pendiente de diseño. A partir de `results` (todas las respuestas característica a
característica) habrá que **consolidar la evidencia** y **deducir el modelo concreto**.

Lo siguiente es un **boceto preliminar, no vinculante**, solo para no perder ideas:

```python
# BOCETO — a redefinir
class Candidate(BaseModel):
    model_name: str          # "Mil Mi-17"
    score: float
    matching: list[str]      # feature_id que apoyan
    contradicting: list[str] # feature_id que contradicen
    reasoning: str

class Report(BaseModel):
    top_candidate: Candidate
    alternatives: list[Candidate]
    evidence_summary: dict   # feature_id → value/reasoning
    markdown: str
```

Decisiones abiertas para la Fase 2:

- ¿Identificación por LLM razonando sobre la evidencia, por *match* contra una tabla de
  referencia por modelo, o un enfoque mixto?
- ¿Cómo se resuelven contradicciones entre características?
- ¿Qué formato tiene el resultado final (JSON, informe legible)?

---

## Orden de Implementación

| Paso | Fichero | Descripción | Fase |
|---|---|---|---|
| 1 | `models.py` | `Feature`, `FeatureResult`, `ExtractionState` | 1 |
| 2 | `config/features.py` | Catálogo de características (la tabla → objetos `Feature`) | 1 |
| 3 | `prompts.py` | `FEATURE_PROMPT_TEMPLATE` | 1 |
| 4 | `client.py` | Wrapper vLLM con forzado de JSON | 1 |
| 5 | `extractor.py` | Recorrido del catálogo: 1 prompt aislado por característica | 1 |
| 6 | `run.py` | CLI: carga imagen, ejecuta extracción, vuelca `results` | 1 |
| — | agregación + identificación | Consolidar evidencia y deducir el modelo | Por definir |
