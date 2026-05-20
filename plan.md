# Vehicle Identifier — Arquitectura del Sistema

## Visión General

Pipeline multimodal basado en LangGraph para identificar modelos de aeronaves
a partir de imágenes, mediante extracción estructurada de características y
razonamiento condicional.

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
    id: str                         # "rotor_blade_count"
    category: str                   # "rotor" | "fuselage" | "tail" | etc.
    question: str                   # Pregunta exacta al LLM
    followup_questions: list[str]   # Preguntas adicionales si confianza < umbral
    required: bool                  # Si es obligatoria para la identificación

class FeatureResult(BaseModel):
    feature_id: str
    value: str                      # Respuesta en lenguaje natural
    structured: dict                # Parsed JSON de la respuesta
    confidence: float               # 0.0 - 1.0
    raw_response: str               # Respuesta completa del LLM

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

### Nodo 5 — `identifier`

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
Analiza esta imagen y responde a la siguiente pregunta sobre la aeronave.
Pregunta: {question}
Responde ÚNICAMENTE con JSON:
{"value": "<descripción concreta>",
 "confidence": <0.0-1.0>,
 "observations": "<detalles adicionales relevantes>"}
"""

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

---

## Ejemplo de Feature (single rotor)

```python
# config/features_single_rotor.py — extracto de 3 features representativas

Feature(
    id="main_rotor_blade_count",
    category="rotor",
    question="¿Cuántas palas tiene el rotor principal de esta aeronave? "
             "Describe también su forma (rectas, escalonadas, en flecha).",
    followup_questions=[
        "Observa el cubo del rotor principal. ¿Puedes confirmar el número exacto de palas?",
        "¿Las palas del rotor tienen doblado en las puntas (blade tip curl)?",
    ],
    required=True,
),
Feature(
    id="tail_rotor_config",
    category="tail",
    question="¿Tiene rotor de cola convencional, fenestron (rotor en túnel), "
             "NOTAR (no tail rotor), o es un diseño coaxial sin rotor de cola?",
    followup_questions=[
        "¿El rotor de cola está montado a izquierda o derecha del estabilizador vertical?",
        "¿Cuántas palas tiene el rotor de cola?",
    ],
    required=True,
),
Feature(
    id="engine_count_position",
    category="propulsion",
    question="¿Cuántos motores tiene y dónde están montados? "
             "(encima del fuselaje, laterales, integrados en la nariz, etc.)",
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
| 9 | `nodes/identifier.py` | Nodo 5 + generación de report |
| 10 | `graph.py` | Ensamblaje final |
| 11 | `run.py` | CLI + tests de integración |
