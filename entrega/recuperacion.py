"""
CODEFEST AD ASTRA 2026 — Etapa 1
recuperacion.py — módulo de recuperación: dada una pregunta, devuelve los 3 documentos y
los 10 fragmentos más relevantes del índice.

Flujo (sección 8.1 de la guía): la pregunta se codifica con el mismo encoder usado para
indexar el corpus (prefijo "query: "), se normaliza, se busca en FAISS, y se arma la
respuesta a nivel de documento (max pooling) y de fragmento (con división de fragmentos
que superen las 250 palabras, respetando oraciones completas).

Componente bonus (opcional): si se le pasa ruta_grafo al construir el Recuperador, además
de buscar por similitud vectorial en FAISS, busca candidatos en el grafo de conocimiento
(grafo_recuperacion.py) y combina las dos listas con Reciprocal Rank Fusion (RRF), tal como
describe la sección 8.5 de la guía. Si no se pasa ruta_grafo, el comportamiento es
exactamente el mismo que sin el grafo — el bonus no cambia nada del sistema base cuando no
se usa.
"""
import json

import faiss

from chunking import dividir_oraciones, _dividir_oracion_larga

PREFIJO_CONSULTA = "query: "
LIMITE_PALABRAS_FRAGMENTO = 250
K_CANDIDATOS_BUSQUEDA = 60
K_CANDIDATOS_PARA_FRAGMENTOS = 15
K_CANDIDATOS_GRAFO = 15
K0_RRF = 60  # constante de suavizado estándar para RRF (sección 8.4 de la guía)


def dividir_en_subfragmentos(texto, limite=LIMITE_PALABRAS_FRAGMENTO):
    oraciones_crudas = dividir_oraciones(texto)
    oraciones = []
    for o in oraciones_crudas:
        oraciones.extend(_dividir_oracion_larga(o, limite=limite))
    if not oraciones:
        return [texto]
    subfragmentos, actual, palabras_actual = [], [], 0
    for oracion in oraciones:
        n = len(oracion.split())
        if actual and palabras_actual + n > limite:
            subfragmentos.append(" ".join(actual))
            actual, palabras_actual = [], 0
        actual.append(oracion)
        palabras_actual += n
    if actual:
        subfragmentos.append(" ".join(actual))
    return subfragmentos


class Recuperador:
    def __init__(self, ruta_index, ruta_metadata, nombre_modelo="intfloat/multilingual-e5-base",
                 ruta_grafo=None, nombre_modelo_ner="Davlan/distilbert-base-multilingual-cased-ner-hrl"):
        self.index = faiss.read_index(str(ruta_index))
        self.metadata = [json.loads(l) for l in open(ruta_metadata, encoding="utf-8")]
        assert self.index.ntotal == len(self.metadata), (
            f"El índice tiene {self.index.ntotal} vectores pero metadata.jsonl tiene "
            f"{len(self.metadata)} líneas; deben coincidir exactamente."
        )
        self._modelo = None
        self._nombre_modelo = nombre_modelo

        self._grafo = None
        if ruta_grafo is not None:
            from grafo_recuperacion import BuscadorGrafo
            self._grafo = BuscadorGrafo(ruta_grafo, nombre_modelo_ner=nombre_modelo_ner)
            self._chunk_por_id = {m["chunk_id"]: m for m in self.metadata}

    def _cargar_modelo(self):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer
            self._modelo = SentenceTransformer(self._nombre_modelo)
        return self._modelo

    def _codificar_consulta(self, texto_pregunta):
        modelo = self._cargar_modelo()
        vector = modelo.encode([PREFIJO_CONSULTA + texto_pregunta], normalize_embeddings=True)
        return vector.astype("float32")

    def buscar_candidatos(self, texto_pregunta, k=K_CANDIDATOS_BUSQUEDA):
        vector = self._codificar_consulta(texto_pregunta)
        similitudes, ids = self.index.search(vector, k)
        candidatos = []
        for score, idx in zip(similitudes[0], ids[0]):
            if idx < 0:
                continue
            registro = self.metadata[idx]
            candidatos.append({
                "doc_id": registro["doc_id"],
                "chunk_id": registro["chunk_id"],
                "texto": registro["texto"],
                "score": float(score),
            })
        return candidatos  # ya vienen ordenados de mayor a menor similitud

    def buscar_candidatos_grafo(self, texto_pregunta, k=K_CANDIDATOS_GRAFO):
        if self._grafo is None:
            return []
        return self._grafo.candidatos(texto_pregunta, self._chunk_por_id, k=k)

    def fusionar_rrf(self, lista_faiss, lista_grafo, k0=K0_RRF):
        """Combina dos listas ya ordenadas de candidatos usando Reciprocal Rank Fusion
        (sección 8.4 de la guía): sRRF(c) = suma sobre los índices donde aparece c de
        1 / (k0 + rango_en_ese_índice(c)). Si un fragmento no aparece en una lista, esa
        lista simplemente no aporta término a su puntaje."""
        puntaje = {}
        info = {}
        for rango, c in enumerate(lista_faiss, start=1):
            puntaje[c["chunk_id"]] = puntaje.get(c["chunk_id"], 0.0) + 1.0 / (k0 + rango)
            info[c["chunk_id"]] = c
        for rango, c in enumerate(lista_grafo, start=1):
            puntaje[c["chunk_id"]] = puntaje.get(c["chunk_id"], 0.0) + 1.0 / (k0 + rango)
            info.setdefault(c["chunk_id"], c)

        fusionados = [dict(info[chunk_id], score=s) for chunk_id, s in puntaje.items()]
        fusionados.sort(key=lambda c: c["score"], reverse=True)
        return fusionados

    def armar_documentos(self, candidatos, top_n=3):
        mejor_por_doc = {}
        for c in candidatos:
            doc_id = c["doc_id"]
            if doc_id not in mejor_por_doc or c["score"] > mejor_por_doc[doc_id]:
                mejor_por_doc[doc_id] = c["score"]
        ordenado = sorted(mejor_por_doc.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"rank": i + 1, "doc_id": doc_id} for i, (doc_id, _score) in enumerate(ordenado)]

    def armar_fragmentos(self, candidatos, top_n=10, k_considerar=K_CANDIDATOS_PARA_FRAGMENTOS):
        piezas = []
        for c in candidatos[:k_considerar]:
            if len(c["texto"].split()) <= LIMITE_PALABRAS_FRAGMENTO:
                piezas.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["texto"]})
            else:
                for sub in dividir_en_subfragmentos(c["texto"]):
                    piezas.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": sub})
            if len(piezas) >= top_n:
                break
        return [{"rank": i + 1, **p} for i, p in enumerate(piezas[:top_n])]

    def responder(self, query_id, texto_pregunta):
        candidatos_faiss = self.buscar_candidatos(texto_pregunta)

        if self._grafo is not None:
            candidatos_grafo = self.buscar_candidatos_grafo(texto_pregunta)
            candidatos = self.fusionar_rrf(candidatos_faiss, candidatos_grafo)
        else:
            candidatos = candidatos_faiss

        return {
            "query_id": query_id,
            "documents": self.armar_documentos(candidatos),
            "fragments": self.armar_fragmentos(candidatos),
        }
