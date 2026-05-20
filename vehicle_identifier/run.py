"""CLI de la Fase 1: carga una imagen, recorre el catálogo y vuelca los resultados.

Ejemplos:
  python -m vehicle_identifier.run foto.jpg
  python -m vehicle_identifier.run foto.jpg --out resultados.json
  python -m vehicle_identifier.run foto.jpg --mock        # sin endpoint, datos simulados
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import DEFAULT_BASE_URL, DEFAULT_MODEL, MockClient, VLLMClient, load_image_b64
from .config.features import HELICOPTER_FEATURES
from .extractor import extract
from .models import ExtractionState


def build_state(image_path: str, *, allow_missing: bool) -> ExtractionState:
    path = Path(image_path)
    if path.exists():
        b64, mime = load_image_b64(image_path)
    elif allow_missing:
        b64, mime = "", "image/jpeg"
    else:
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    return ExtractionState(
        image_path=image_path,
        image_b64=b64,
        image_mime=mime,
        features=list(HELICOPTER_FEATURES),
    )


def state_to_dict(state: ExtractionState) -> dict:
    return {
        "image_path": state.image_path,
        "n_features": len(state.features),
        "results": [
            {
                "feature_id": r.feature_id,
                "value": r.value,
                "reasoning": r.reasoning,
                "error": r.error,
            }
            for r in state.results
        ],
        "errors": state.errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extracción de características (Fase 1)")
    parser.add_argument("image", help="Ruta de la imagen del vehículo")
    parser.add_argument("--out", help="Ruta donde guardar el JSON de resultados")
    parser.add_argument("--mock", action="store_true", help="Usa un cliente simulado (sin endpoint)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="URL del endpoint vLLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Nombre del modelo")
    parser.add_argument("--force-json", action="store_true", help="Pide response_format JSON al endpoint")
    args = parser.parse_args(argv)

    state = build_state(args.image, allow_missing=args.mock)

    if args.mock:
        client = MockClient()
    else:
        client = VLLMClient(base_url=args.base_url, model=args.model, force_json=args.force_json)

    state = extract(state, client)

    output = json.dumps(state_to_dict(state), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Resultados escritos en {args.out}", file=sys.stderr)
    print(output)

    return 1 if state.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
