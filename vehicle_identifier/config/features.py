"""Catálogo de características (trabajo humano).

Catálogo SEMILLA para helicópteros. El experto debe ampliarlo/ajustarlo:
cada entrada es una característica que el flujo preguntará de forma aislada.
"""

from __future__ import annotations

from ..models import Feature

HELICOPTER_FEATURES: list[Feature] = [
    Feature(
        id="main_rotor_blade_count",
        category="rotor",
        question="¿Cuántas palas tiene el rotor principal y qué forma/anchura tienen?",
        context=(
            "Cuenta las palas ancladas al cubo del rotor principal (eje vertical sobre "
            "el fuselaje); ignora el rotor de cola. Fíjate en el ancho relativo de la "
            "pala y en si la punta es recta, en flecha o doblada."
        ),
        examples=[
            "3 palas rectas y estrechas",
            "5 palas anchas con punta en flecha",
            "palas con doblez hacia abajo en la punta (blade tip curl)",
        ],
    ),
    Feature(
        id="tail_rotor_config",
        category="tail",
        question="¿Qué tipo de rotor de cola tiene?",
        context=(
            "Mira la parte trasera, en la base del estabilizador vertical. Distingue "
            "entre rotor de cola convencional expuesto, fenestron (rotor carenado dentro "
            "de un conducto), NOTAR (sin rotor visible, salida de aire en el botalón) o "
            "diseño coaxial (sin rotor de cola, dos rotores principales contrarrotativos)."
        ),
        examples=[
            "rotor de cola convencional de 2-4 palas expuesto",
            "fenestron (rotor embutido en el carenado de la deriva)",
            "NOTAR: botalón liso sin rotor de cola",
        ],
    ),
    Feature(
        id="engine_count_position",
        category="propulsion",
        question="¿Cuántos motores tiene y dónde están montados?",
        context=(
            "Localiza los carenados de motor y las tomas/escapes. En helicópteros suelen "
            "ir sobre el fuselaje, a los lados de la transmisión principal o integrados en "
            "la nariz. Cuenta tomas de aire y toberas de escape diferenciadas."
        ),
        examples=[
            "1 motor sobre el fuselaje con escape lateral",
            "2 motores laterales a ambos lados del rotor principal",
            "2 motores con tomas de aire sobre la cabina",
        ],
    ),
    Feature(
        id="landing_gear_type",
        category="landing_gear",
        question="¿Qué tren de aterrizaje tiene: patines o ruedas?",
        context=(
            "Mira la parte inferior del fuselaje. Distingue patines fijos (skids) de tren "
            "de ruedas; si son ruedas, fíjate en si parecen fijas o retráctiles y en el "
            "número de puntos de apoyo."
        ),
        examples=[
            "patines fijos (skids)",
            "tren triciclo de ruedas, aparentemente retráctil",
            "ruedas fijas con rueda de cola",
        ],
    ),
    Feature(
        id="cockpit_glazing",
        category="fuselage",
        question="¿Cómo es la cabina/acristalamiento de la cabina de pilotaje?",
        context=(
            "Observa el morro y el puesto de pilotaje. Distingue cabina en tándem (pilotos "
            "uno detrás de otro, típica de helicópteros de ataque) de cabina lado a lado, "
            "y fíjate en la forma del acristalamiento (escalonado, redondeado, facetado)."
        ),
        examples=[
            "cabina en tándem escalonada (ataque)",
            "cabina lado a lado con parabrisas redondeado",
            "morro acristalado amplio tipo observación",
        ],
    ),
    Feature(
        id="stub_wings_armament",
        category="mission",
        question="¿Tiene alas cortas (stub wings) o puntos de anclaje con armamento o cargas?",
        context=(
            "Mira los laterales del fuselaje a la altura de la cabina/transmisión. Busca "
            "alas cortas con soportes (pylons), lanzacohetes, misiles, depósitos o sensores. "
            "Su ausencia sugiere uso civil o de transporte."
        ),
        examples=[
            "alas cortas con lanzacohetes y misiles",
            "sin alas ni soportes (configuración civil)",
            "soportes laterales con depósitos auxiliares",
        ],
    ),
    Feature(
        id="fuselage_silhouette",
        category="fuselage",
        question="¿Cuál es la silueta general del fuselaje?",
        context=(
            "Valora la proporción y el volumen del fuselaje: esbelto y estrecho (ataque), "
            "voluminoso con cabina de carga amplia (transporte), o compacto (ligero/utilitario). "
            "Fíjate en rampa trasera o puertas de carga grandes."
        ),
        examples=[
            "fuselaje esbelto y estrecho (ataque)",
            "fuselaje voluminoso con rampa trasera (transporte)",
            "fuselaje compacto ligero (utilitario)",
        ],
    ),
]
