# Vehicle Identifier — Arquitectura del Sistema

## Visión General

Sistema para identificar vehículos a partir de imágenes (objetivo inicial:
**helicópteros**) descomponiendo la observación en **características discriminantes**
curadas por un experto humano. En lugar de preguntar al modelo "¿qué helicóptero es?"
de forma abierta, el sistema interroga al modelo multimodal **una característica a la
vez**, con una pregunta concreta y contexto que guía en qué fijarse.

### Reparto de trabajo

- **Trabajo humano (conocimiento experto, offline):** definir, por tipo de vehículo,
  el catálogo de características a reconocer. Por cada característica se aporta: la
  **pregunta** concreta, el **contexto** (en qué fijarse al mirar la imagen) y
  **ejemplos** concretos de lo que se busca identificar.
- **Trabajo autónomo (runtime):** un flujo que recorre ese catálogo y, por cada
  característica, construye **un único prompt** (pregunta + contexto + ejemplos), lo
  envía al modelo multimodal y guarda la respuesta **validada característica por
  característica**.

### Hipótesis de diseño

`Gemma-4` detecta rasgos concretos con alta fiabilidad cuando la pregunta es
**estrecha y va acompañada de ejemplos** de qué buscar. Preguntar característica a
característica (con su contexto) produce una validación más fiable, auditable y
controlable por el experto que una sola pregunta abierta de identificación.

### Fases

| Fase | Alcance | Estado |
|---|---|---|
| **1 — Extracción de características** | Catálogo humano + flujo que pregunta una característica por prompt y recoge respuestas validadas | **Foco actual** |
| **2 — Identificación** | Razonar sobre la evidencia agregada para proponer el/los modelo(s) | **Posterior — a definir** |

> El diseño de la **Fase 2** (nodo `identifier`, schemas `Candidate`/`Report` y
> `IDENTIFY_PROMPT_TEMPLATE` que aparecen más abajo) es un **boceto preliminar**: se
> redefinirá cuando se aborde esa fase. La especificación firme de este documento es
> la **Fase 1**.

---

## Stack Técnico

| Componente | Librería | Motivo |
|---|---|---|
| Orquestación | `langgraph` | Grafo con edges condicionales |
| Schema / validación | `pydantic v2` | Estado tipado, parsing de respuestas JSON |
| LLM client | `openai` | Compatible con vLLM endpoint local |
| Modelo | `nvidia/Gemma-4-31B-IT-NVFP4` | Multimodal, local, ya operativo |

```bash
pip install langgraph pydantic openai
```

---

## Estado Global del Grafo

```python
# models.py
class AircraftState(BaseModel):
    # Input
    image_path: str
    image_b64: str                          # cargado al inicio, reutilizado

    # Clasificación inicial
    aircraft_class: str | None              # "single_rotor" | "tandem" | "fixed_wing" | "multirotor"
    classification_confidence: float        # 0.0 - 1.0

    # Feature extraction
    feature_queue: list[Feature]            # preguntas pendientes de la rama activa
    features_done: list[FeatureResult]      # respuestas completadas
    low_confidence_flags: list[str]         # IDs de features con confianza < umbral

    # Follow-up
    followup_queue: list[Feature]           # preguntas de seguimiento generadas condicionalmente
    followup_done: list[FeatureResult]      # respuestas de seguimiento

    # Identificación
    candidates: list[Candidate]            # top-3 modelos con score
    final_report: Report | None            # output final

    # Control
    iteration_count: int                   # salvaguarda anti-loop
    errors: list[str]                      # errores no fatales acumulados
```

---

## Estructura de Ficheros

```
vehicle_identifier/
│
├── config/
│   ├── __init__.py
│   ├── features_single_rotor.py    # 20 Feature definitions para rotor simple
│   ├── features_tandem.py          # 20 Feature definitions para tandem (Chinook-like)
│   ├── features_fixed_wing.py      # 20 Feature definitions para ala fija
│   └── features_multirotor.py      # 15 Feature definitions para multirotor
│
├── nodes/
│   ├── __init__.py
│   ├── classifier.py               # Nodo 1: detecta clase de aeronave
│   ├── extractor.py                # Nodo 2: ejecuta feature queries en loop
│   ├── checker.py                  # Nodo 3: evalúa confianza, genera follow-ups
│   ├── aggregator.py               # Nodo 4: consolida evidencia
│   └── identifier.py               # Nodo 5: identificación final + report
│
├── models.py                       # Pydantic schemas: State, Feature, Result, Report
├── client.py                       # Wrapper del cliente vLLM (reutiliza _test_.py logic)
├── graph.py                        # Definición y compilación del grafo LangGraph
├── prompts.py                      # Todos los prompts centralizados
└── run.py                          # Entry point CLI
```

---

## Schemas Pydantic

```python
# models.py — schemas clave

class Feature(BaseModel):
    id: str                         # "main_rotor_blade_count"
    category: str                   # "rotor" | "fuselage" | "tail" | etc.
    question: str                   # Pregunta concreta sobre ESTA característica
    context: str                    # Guía de en qué fijarse al observar la imagen
    examples: list[str]             # Ejemplos concretos de lo que se busca / valores posibles
    followup_questions: list[str]   # Preguntas adicionales si confianza < umbral
    required: bool                  # Si es obligatoria para la identificación

class FeatureResult(BaseModel):
    feature_id: str
    value: str                      # Respuesta en lenguaje natural
    structured: dict                # Parsed JSON de la respuesta
    confidence: float               # 0.0 - 1.0
    raw_response: str               # Respuesta completa del LLM

# --- FASE 2 (boceto preliminar — a redefinir): identificación ---
class Candidate(BaseModel):
    model_name: str                 # "Mil Mi-17"
    score: float                    # 0.0 - 1.0
    matching_features: list[str]    # IDs de features que soportan este candidato
    contradicting_features: list[str]
    reasoning: str

class Report(BaseModel):
    top_candidate: Candidate
    alternatives: list[Candidate]   # top 2-3
    confidence_overall: float
    evidence_summary: dict          # feature_id → valor clave
    low_confidence_warnings: list[str]
    markdown: str                   # Informe legible generado
```

---

## Nodos del Grafo

### Nodo 1 — `classifier`

**Entrada:** `image_b64`
**Salida:** `aircraft_class`, `classification_confidence`

- Una sola llamada multimodal
- Pregunta abierta: identifica el tipo general de aeronave
- Fuerza respuesta en JSON: `{"class": "...", "confidence": 0.x, "reasoning": "..."}`
- Carga la `feature_queue` correspondiente a la clase detectada

**Edge condicional de salida:**
```
aircraft_class == "single_rotor"  → extractor (con features_single_rotor)
aircraft_class == "tandem"        → extractor (con features_tandem)
aircraft_class == "fixed_wing"    → extractor (con features_fixed_wing)
aircraft_class == "multirotor"    → extractor (con features_multirotor)
aircraft_class == "unknown"       → END (con error)
```

---

### Nodo 2 — `extractor`

**Entrada:** `feature_queue`, `image_b64`
**Salida:** `features_done`, `low_confidence_flags`

- Itera sobre `feature_queue` (10-25 preguntas según la rama)
- Una llamada multimodal por Feature
- Cada llamada fuerza respuesta JSON con schema `FeatureResult`
- Acumula resultados en `features_done`
- Marca en `low_confidence_flags` cualquier resultado con `confidence < 0.6`

**Edge condicional de salida:**
```
len(low_confidence_flags) > 0  → checker
len(low_confidence_flags) == 0 → aggregator
```

---

### Nodo 3 — `checker`

**Entrada:** `low_confidence_flags`, `features_done`, `image_b64`
**Salida:** `followup_queue` poblado

- Para cada feature con baja confianza, carga sus `followup_questions`
- Opcionalmente genera follow-ups dinámicos adicionales via LLM
- Limita a `MAX_FOLLOWUPS = 10` para evitar loops
- Controla `iteration_count` como salvaguarda

**Edge condicional de salida:**
```
followup_queue no vacía  → followup_extractor (variante de extractor para follow-ups)
followup_queue vacía     → aggregator
```

---

### Nodo 3b — `followup_extractor`

**Entrada:** `followup_queue`, `image_b64`
**Salida:** `followup_done`

- Idéntico a `extractor` pero opera sobre `followup_queue`
- Los resultados se guardan en `followup_done` (no sobreescriben `features_done`)
- Siempre va a `aggregator` al terminar (sin más bifurcaciones)

---

### Nodo 4 — `aggregator`

**Entrada:** `features_done`, `followup_done`
**Salida:** Estado enriquecido listo para identificación

- Fusiona `features_done` + `followup_done`
- Resuelve contradicciones: si dos features dan valores incompatibles, marca ambas
- Calcula confianza media global
- Construye `evidence_summary` como dict compacto para el prompt de identificación

**Edge:** siempre → `identifier`

---

### Nodo 5 — `identifier`  *(FASE 2 — boceto preliminar, a redefinir)*

**Entrada:** `evidence_summary`, `aircraft_class`, `image_b64`
**Salida:** `candidates`, `final_report`

- Una sola llamada con toda la evidencia agregada en el prompt
- Pide top-3 candidatos con score y justificación feature por feature
- Genera el `Report` final incluyendo markdown legible
- Persiste el resultado en JSON + `.md`

**Edge:** siempre → `END`

---

## Grafo LangGraph

```python
# graph.py — estructura del grafo

from langgraph.graph import StateGraph, END

def build_graph():
    graph = StateGraph(AircraftState)

    graph.add_node("classifier",         classifier_node)
    graph.add_node("extractor",          extractor_node)
    graph.add_node("checker",            checker_node)
    graph.add_node("followup_extractor", followup_extractor_node)
    graph.add_node("aggregator",         aggregator_node)
    graph.add_node("identifier",         identifier_node)

    graph.set_entry_point("classifier")

    # classifier → rama según tipo de aeronave
    graph.add_conditional_edges("classifier", route_by_class, {
        "single_rotor": "extractor",
        "tandem":       "extractor",
        "fixed_wing":   "extractor",
        "multirotor":   "extractor",
        "unknown":      END,
    })

    # extractor → checker si hay baja confianza, si no directo a aggregator
    graph.add_conditional_edges("extractor", route_by_confidence, {
        "needs_followup": "checker",
        "ok":             "aggregator",
    })

    # checker → followup_extractor si hay follow-ups, si no directo a aggregator
    graph.add_conditional_edges("checker", route_by_followup, {
        "has_followups": "followup_extractor",
        "skip":          "aggregator",
    })

    graph.add_edge("followup_extractor", "aggregator")
    graph.add_edge("aggregator",         "identifier")
    graph.add_edge("identifier",         END)

    return graph.compile()
```

---

## Prompts — Estructura

```python
# prompts.py

CLASSIFY_PROMPT = """
Analiza esta imagen y clasifica la aeronave. Responde ÚNICAMENTE con JSON:
{"class": "<single_rotor|tandem|fixed_wing|multirotor|unknown>",
 "confidence": <0.0-1.0>,
 "reasoning": "<breve justificación>"}
"""

FEATURE_PROMPT_TEMPLATE = """
Analiza esta imagen y céntrate EXCLUSIVAMENTE en una característica del vehículo.

Característica: {question}

En qué fijarte: {context}

Ejemplos de lo que se busca identificar:
{examples}

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{{"value": "<descripción concreta de lo observado>",
  "confidence": <0.0-1.0>,
  "observations": "<detalles relevantes; indica si no hay evidencia suficiente>"}}
"""

# FASE 2 (boceto preliminar — a redefinir cuando se aborde la identificación)
IDENTIFY_PROMPT_TEMPLATE = """
Eres un experto en identificación de aeronaves militares y civiles.
A continuación tienes las características extraídas de una aeronave:

Clase detectada: {aircraft_class}
Evidencia:
{evidence_json}

Identifica el modelo exacto o los modelos más probables. Responde ÚNICAMENTE con JSON:
{"candidates": [
    {{"model": "<fabricante modelo variante>",
      "score": <0.0-1.0>,
      "matching": ["<feature_id>", ...],
      "contradicting": ["<feature_id>", ...],
      "reasoning": "<cadena de razonamiento>"}},
    ...
  ],
  "overall_confidence": <0.0-1.0>
}
"""
```

> **Nota sobre los prompts.** `FEATURE_PROMPT_TEMPLATE` se rellena con `str.format()`,
> por lo que las llaves literales del JSON van escapadas como `{{ }}`. La variable
> `{examples}` se renderiza como lista (un ejemplo por línea) antes de formatear. El
> forzado de JSON debe apoyarse en la salida estructurada del endpoint (vLLM
> `guided_json` / `response_format`), no solo en pedirlo en el texto.

---

## Ejemplo de Feature (single rotor)

```python
# config/features_single_rotor.py — extracto de 3 features representativas

Feature(
    id="main_rotor_blade_count",
    category="rotor",
    question="¿Cuántas palas tiene el rotor principal y qué forma/anchura tienen?",
    context="Cuenta las palas ancladas al cubo del rotor principal (eje vertical sobre "
            "el fuselaje); ignora el rotor de cola. Fíjate en el ancho relativo de la "
            "pala y en si la punta es recta, en flecha o doblada.",
    examples=[
        "3 palas rectas y estrechas",
        "5 palas anchas con punta en flecha",
        "palas con doblez hacia abajo en la punta (blade tip curl)",
    ],
    followup_questions=[
        "Observa el cubo del rotor principal. ¿Puedes confirmar el número exacto de palas?",
        "¿Las palas del rotor tienen doblado en las puntas (blade tip curl)?",
    ],
    required=True,
),
Feature(
    id="tail_rotor_config",
    category="tail",
    question="¿Qué tipo de rotor de cola tiene?",
    context="Mira la parte trasera, en la base del estabilizador vertical. Distingue "
            "entre rotor de cola convencional expuesto, fenestron (rotor carenado dentro "
            "de un conducto), NOTAR (sin rotor visible, salida de aire en el botalón) o "
            "diseño coaxial (sin rotor de cola, dos rotores principales contrarrotativos).",
    examples=[
        "rotor de cola convencional de 2-4 palas expuesto",
        "fenestron (rotor embutido en el carenado de la deriva)",
        "NOTAR: botalón liso sin rotor de cola",
    ],
    followup_questions=[
        "¿El rotor de cola está montado a izquierda o derecha del estabilizador vertical?",
        "¿Cuántas palas tiene el rotor de cola?",
    ],
    required=True,
),
Feature(
    id="engine_count_position",
    category="propulsion",
    question="¿Cuántos motores tiene y dónde están montados?",
    context="Localiza los carenados de motor y las tomas/escapes. En helicópteros suelen "
            "ir sobre el fuselaje, a los lados de la transmisión principal o integrados en "
            "la nariz. Cuenta tomas de aire y toberas de escape diferenciadas para inferir "
            "el número.",
    examples=[
        "1 motor sobre el fuselaje con escape lateral",
        "2 motores laterales a ambos lados del rotor principal",
        "2 motores con tomas de aire sobre la cabina",
    ],
    followup_questions=[
        "¿Los escapes de los motores son laterales, hacia arriba o hacia atrás?",
        "¿Se aprecian tomas de aire diferenciadas o son integrales con la carcasa?",
    ],
    required=True,
),
```

---

## Orden de Implementación

| Paso | Fichero(s) | Descripción |
|---|---|---|
| 1 | `models.py` | Schemas Pydantic completos |
| 2 | `config/features_*.py` | Las 4 listas de features (10-20 por rama) |
| 3 | `prompts.py` | Todos los prompts |
| 4 | `client.py` | Wrapper vLLM con JSON forcing |
| 5 | `nodes/classifier.py` | Nodo 1 |
| 6 | `nodes/extractor.py` | Nodo 2 + 3b |
| 7 | `nodes/checker.py` | Nodo 3 con lógica condicional |
| 8 | `nodes/aggregator.py` | Nodo 4 |
| 9 | `nodes/identifier.py` | Nodo 5 + generación de report — **Fase 2** |
| 10 | `graph.py` | Ensamblaje final |
| 11 | `run.py` | CLI + tests de integración |
