"""
CODEFEST AD ASTRA 2026 — Etapa 1 (componente bonus)
grafo_recuperacion.py — búsqueda de candidatos usando el grafo de conocimiento.

Este módulo es independiente de recuperacion.py a propósito: si algo falla acá, o si el
grafo todavía no está construido, el sistema de búsqueda por vectores (FAISS) sigue
funcionando exactamente igual, sin depender de esto.

Qué hace BuscadorGrafo.candidatos(texto_pregunta):
1. Detecta las entidades mencionadas en la pregunta con el mismo modelo NER usado para
   construir el grafo (Davlan/distilbert-base-multilingual-cased-ner-hrl).
2. Para cada entidad de la pregunta que existe en el grafo, reúne los fragmentos donde esa
   entidad aparece directamente, y los fragmentos asociados a sus vecinos de primer orden
   (entidades relacionadas), tal como describe la sección 8.5 de la guía oficial.
3. Devuelve una lista ordenada de fragmentos candidatos, en el mismo formato que usa
   recuperacion.py para los candidatos de FAISS, lista para fusionarse con esos resultados.

No usa ningún modelo generativo: el NER es un modelo de clasificación de tokens (encoder),
y el resto es aritmética simple sobre el grafo.

--- NOTA SOBRE CARGA (v2) ---
El grafo de este corpus terminó pesando más de 2 GB en formato GraphML (texto XML), y
leer un XML de ese tamaño con nx.read_graphml() es extremadamente lento. Por eso, si existe
un "checkpoint.pkl" en la misma carpeta que el grafo.graphml (construir_grafo.py lo deja ahí
automáticamente al terminar, con los mismos datos en formato binario), se usa ese en su
lugar -- se carga en segundos en vez de minutos. El grafo.graphml sigue siendo el archivo
"oficial" que se entrega; esto solo cambia cómo lo lee este módulo internamente.
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx


class BuscadorGrafo:
    def __init__(self, ruta_grafo, nombre_modelo_ner="Davlan/distilbert-base-multilingual-cased-ner-hrl"):
        ruta_grafo = Path(ruta_grafo)
        ruta_pickle = ruta_grafo.parent / "checkpoint.pkl"
        if ruta_pickle.exists():
            with open(ruta_pickle, "rb") as f:
                self.grafo = pickle.load(f)
        else:
            self.grafo = nx.read_graphml(str(ruta_grafo))
        self._nombre_modelo_ner = nombre_modelo_ner
        self._ner = None

    def _cargar_ner(self):
        if self._ner is None:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            tokenizer = AutoTokenizer.from_pretrained(self._nombre_modelo_ner)
            modelo = AutoModelForTokenClassification.from_pretrained(self._nombre_modelo_ner)
            self._ner = pipeline("ner", model=modelo, tokenizer=tokenizer, aggregation_strategy="simple")
        return self._ner

    @staticmethod
    def _normalizar(texto):
        return " ".join(texto.split()).strip(" .,;:()[]\"'-").upper()

    def entidades_de(self, texto_pregunta):
        ner = self._cargar_ner()
        resultado = ner(texto_pregunta)
        vistas, entidades = set(), []
        for ent in resultado:
            nombre = self._normalizar(ent["word"])
            if len(nombre) < 2 or nombre in vistas:
                continue
            vistas.add(nombre)
            entidades.append(nombre)
        return entidades

    def _evidencia(self, texto_ejemplos):
        pares = []
        for e in texto_ejemplos.split(";"):
            if not e or ":" not in e:
                continue
            doc_id, chunk_id = e.split(":", 1)
            pares.append((doc_id, chunk_id))
        return pares

    def candidatos(self, texto_pregunta, chunk_por_id, k=15, max_vecinos_por_entidad=30):
        """
        chunk_por_id: diccionario chunk_id -> registro de metadata (con al menos 'texto' y 'doc_id'),
        para poder devolver el texto real de cada fragmento candidato.

        max_vecinos_por_entidad: algunas entidades (países, organizaciones muy nombradas)
        terminan conectadas a miles de vecinos distintos en un grafo de este tamaño. Si se
        usaran TODOS esos vecinos, una entidad muy genérica ahogaría los resultados con
        fragmentos poco relacionados con la pregunta real. Por eso solo se usan los
        "max_vecinos_por_entidad" vecinos con mayor peso (los más fuertemente conectados),
        que son los que de verdad aportan señal.
        """
        entidades_query = [e for e in self.entidades_de(texto_pregunta) if e in self.grafo]
        if not entidades_query:
            return []

        puntaje_por_chunk = defaultdict(float)

        for entidad in entidades_query:
            # 1. Menciones directas de la entidad
            ejemplos_nodo = self.grafo.nodes[entidad].get("ejemplos", "")
            for doc_id, chunk_id in self._evidencia(ejemplos_nodo):
                puntaje_por_chunk[chunk_id] += 1.0

            # 2. Vecinos de primer orden (entidades relacionadas), limitado a los más fuertes
            vecinos = self.grafo[entidad]
            vecinos_ordenados = sorted(
                vecinos.items(), key=lambda par: float(par[1].get("peso", 1)), reverse=True
            )[:max_vecinos_por_entidad]
            for vecino, datos_arista in vecinos_ordenados:
                peso = datos_arista.get("peso", 1)
                ejemplos_arista = datos_arista.get("ejemplos", "")
                for doc_id, chunk_id in self._evidencia(ejemplos_arista):
                    puntaje_por_chunk[chunk_id] += float(peso)

        candidatos_ordenados = sorted(puntaje_por_chunk.items(), key=lambda x: x[1], reverse=True)

        resultado = []
        for chunk_id, puntaje in candidatos_ordenados:
            registro = chunk_por_id.get(chunk_id)
            if registro is None:
                continue  # fragmento de evidencia que ya no existe en la metadata actual
            resultado.append({
                "doc_id": registro["doc_id"],
                "chunk_id": chunk_id,
                "texto": registro["texto"],
                "score": puntaje,
            })
            if len(resultado) >= k:
                break
        return resultado
