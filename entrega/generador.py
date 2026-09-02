"""
CODEFEST AD ASTRA 2026 — Etapa 1
generador.py — script reproducible de la entrega.

Qué hace: lee el índice YA CONSTRUIDO (base_vectorial/encoder_multilingual-e5-base/) y las
50 preguntas oficiales de evaluación, y genera resultados.jsonl con el formato exacto que
exige la guía oficial: 50 líneas (q001-q050), cada una con 3 documentos + 10 fragmentos.

Importante (regla de la guía): este script NO reentrena ni reconstruye el índice desde el
corpus -- solo reproduce la búsqueda sobre lo que ya está construido. Debe poder correr en
cualquier máquina con Python y las librerías instaladas, sin rutas absolutas ni nada
específico de un computador en particular (por eso todas las rutas de abajo son relativas).

Componente bonus: si existe grafo/grafo.graphml, se usa también el grafo de conocimiento
para buscar candidatos y se combinan con los resultados de FAISS (sección 8.5 de la guía).
Si esa carpeta no existe, el sistema funciona exactamente igual que sin el grafo -- este
bonus no es una condición para que el script funcione.
"""

import json
from pathlib import Path

from recuperacion import Recuperador

RUTA_INDEX = "base_vectorial/encoder_multilingual-e5-base/index.faiss"
RUTA_METADATA = "base_vectorial/encoder_multilingual-e5-base/metadata.jsonl"
RUTA_PREGUNTAS = "preguntas_evaluacion_50.jsonl"
RUTA_SALIDA = "resultados.jsonl"
RUTA_GRAFO = "base_vectorial/grafo/grafo.graphml"


def cargar_preguntas(ruta):
    preguntas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                preguntas.append(json.loads(linea))
    return preguntas


def main():
    for ruta in (RUTA_INDEX, RUTA_METADATA, RUTA_PREGUNTAS):
        if not Path(ruta).exists():
            raise FileNotFoundError(
                f"No se encontró '{ruta}'. Corre este script desde la carpeta que contiene "
                f"base_vectorial/ y preguntas_evaluacion_50.jsonl."
            )

    preguntas = cargar_preguntas(RUTA_PREGUNTAS)
    print(f"Preguntas cargadas: {len(preguntas)}")
    if len(preguntas) != 50:
        print(f"AVISO: se esperaban 50 preguntas y se cargaron {len(preguntas)}.")

    usar_grafo = Path(RUTA_GRAFO).exists()
    if usar_grafo:
        print(f"Grafo de conocimiento encontrado en '{RUTA_GRAFO}': se usará junto con FAISS.")
    else:
        print("No se encontró grafo de conocimiento: se usará solo la búsqueda vectorial (FAISS).")

    print("Cargando índice y modelo (puede tardar un poco la primera vez)...")
    rec = Recuperador(RUTA_INDEX, RUTA_METADATA, ruta_grafo=RUTA_GRAFO if usar_grafo else None)

    with open(RUTA_SALIDA, "w", encoding="utf-8") as f_out:
        for i, p in enumerate(preguntas, start=1):
            resultado = rec.responder(p["query_id"], p["text"])
            assert len(resultado["documents"]) == 3, f"{p['query_id']}: se esperaban 3 documentos"
            assert len(resultado["fragments"]) == 10, f"{p['query_id']}: se esperaban 10 fragmentos"
            for frag in resultado["fragments"]:
                assert len(frag["text"].split()) <= 250, f"{p['query_id']}: fragmento supera 250 palabras"
            f_out.write(json.dumps(resultado, ensure_ascii=False) + "\n")
            print(f"  ...{i}/{len(preguntas)} -> {p['query_id']} OK", flush=True)

    print(f"Listo. '{RUTA_SALIDA}' generado con {len(preguntas)} líneas.")


if __name__ == "__main__":
    main()
