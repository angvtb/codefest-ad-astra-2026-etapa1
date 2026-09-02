# CODEFEST AD ASTRA 2026 — Etapa 1: Sistema de Recuperación Semántica

**Equipo:** The Data Alchemists — Universidad Icesi

## Qué hace este proyecto

Dado un corpus de ~1.826 documentos (PDF, JSON, CSV, Excel, imágenes) sobre tres frentes temáticos —IA en el ámbito militar, seguridad del entorno espacial y dinámicas territoriales en América Latina—, el sistema recibe una pregunta en lenguaje natural y devuelve los 3 documentos y 10 fragmentos de texto más relevantes, usando únicamente búsqueda semántica vectorial: sin modelos generativos en ningún punto del proceso, tal como lo exigía la convocatoria.

## Arquitectura

```
Documentos → Extracción y limpieza → Chunking → Embeddings (encoder) → Índice FAISS
                                                                              ↓
                                                    Grafo de conocimiento (bonus) → RRF
                                                                              ↓
                                                              generador.py → resultados.jsonl
```

- **Extracción**: `pypdf` (PDF), parsing nativo (JSON/CSV), `openpyxl` (Excel), Tesseract OCR (imágenes escaneadas).
- **Chunking**: fragmentación por oraciones completas, ~250 palabras objetivo, tope duro de 375 palabras, con solape de una oración entre fragmentos consecutivos.
- **Encoder**: [`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) (768 dimensiones, licencia MIT, soporte multilingüe ES/EN/PT).
- **Índice vectorial**: FAISS `IndexFlatIP` (similitud coseno vía producto interno normalizado).
- **Grafo de conocimiento** (componente bonus): entidades extraídas con `Davlan/distilbert-base-multilingual-cased-ner-hrl`, relaciones por co-ocurrencia, fusionado con los resultados de FAISS mediante Reciprocal Rank Fusion (RRF).

## Resultados finales

| Métrica | Valor |
|---|---|
| Documentos procesados | 1.671 de 1.826 (155 excluidos y documentados) |
| Fragmentos indexados | 126.687 |
| Entidades en el grafo | 414.942 |
| Relaciones en el grafo | 11.043.918 |

## Estructura del repositorio

```
entrega/
  resultados.jsonl              # Salida sobre las 50 preguntas de evaluación oficiales
  generador.py                  # Script reproducible: preguntas → resultados.jsonl
  informe_tecnico.pdf           # Justificación de chunking, encoder, índice y grafo
  recuperacion.py               # Búsqueda vectorial (FAISS)
  chunking.py                   # Fragmentación de documentos
  grafo_recuperacion.py         # Búsqueda sobre el grafo de conocimiento
  preguntas_evaluacion_50.jsonl # Preguntas oficiales de evaluación
  base_vectorial/
    encoder_multilingual-e5-base/
      index.faiss
      metadata.jsonl
    grafo/
      grafo.graphml
```

## Reproducir los resultados

```bash
cd entrega/
pip install -r requirements.txt   # faiss-cpu, sentence-transformers, networkx, openpyxl, pypdf
python generador.py
```

El script valida automáticamente que cada respuesta tenga exactamente 3 documentos y 10 fragmentos, ninguno mayor a 250 palabras.

## Procedencia y licencia de los datos

El corpus fue construido por la organización a partir de fuentes abiertas. Este repositorio se publica como material de portafolio académico, en línea con las opciones de entrega que la organización habilitó explícitamente para el reto.

## Licencia del código

El código de este repositorio se publica bajo licencia MIT (ver `LICENSE`).

## Contribuyentes

- Daniel Stiven Alarcón Ijají
- Frank Camilo Mendoza Jiménez
- Angie Valentina Támara Becerra
- Edward Duván Atehortúa Sánchez
