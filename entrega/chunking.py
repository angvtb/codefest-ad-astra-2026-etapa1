"""
CODEFEST AD ASTRA 2026 — Etapa 1
Chunking: corta cada documento de documentos_extraidos.jsonl en fragmentos ("chunks").

Reglas de diseño (ver notas_codefest_2026.md):
- Nunca cortar una oración a la mitad (requisito obligatorio de la guía oficial).
- Tamaño de referencia: ~250 palabras por fragmento.
- Solapamiento entre fragmentos consecutivos (se repite la última oración del fragmento
  anterior al inicio del siguiente) para no perder contexto en el corte.
- Metadata obligatoria por fragmento (Tabla 1): doc_id, chunk_id, fuente, formato, fenomeno,
  posicion, num_tokens, texto.
- `num_tokens` por ahora es un conteo de PALABRAS (aproximación) -- se recalcula con el
  tokenizador real del encoder una vez elegido (Tarea #7/#8).

Entrada: documentos_extraidos.jsonl (salida de extraer_corpus.py)
Salida: chunks.jsonl (un fragmento por línea)
"""

import json
import re
import sys
from pathlib import Path

TAMANO_OBJETIVO_PALABRAS = 250
SOLAPAMIENTO_ORACIONES = 1  # cuántas oraciones del final de un chunk se repiten al inicio del siguiente

# Abreviaturas comunes (ES/EN) que no deben interpretarse como fin de oración.
ABREVIATURAS = {
    "sr.", "sra.", "srta.", "dr.", "dra.", "ee.uu.", "u.s.", "u.k.", "vs.", "etc.",
    "art.", "arts.", "núm.", "no.", "pp.", "p.", "vol.", "cap.", "fig.", "ej.",
    "mr.", "mrs.", "ms.", "prof.", "gral.", "cnel.", "cor.", "gen.", "col.",
    "inc.", "ltd.", "co.", "corp.", "depto.", "dept.", "gob.", "int.",
}


def dividir_oraciones(texto):
    """Separa un texto en oraciones completas, evitando cortar por abreviaturas comunes."""
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return []
    partes = re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡"“])', texto)

    oraciones = []
    buffer = ""
    for parte in partes:
        candidato = f"{buffer} {parte}".strip() if buffer else parte
        ultima_palabra = candidato.split(" ")[-1].lower() if candidato else ""
        if ultima_palabra in ABREVIATURAS:
            buffer = candidato
        else:
            oraciones.append(candidato)
            buffer = ""
    if buffer:
        oraciones.append(buffer.strip())

    # Red de seguridad: texto real (listados legales, tablas, leyendas con muchas abreviaturas
    # tipo "Inc.", "Corp.", "S.A.") a veces no tiene una frontera de oración clara por buen
    # tramo y el detector de arriba junta todo en un solo bloque gigante. Si una "oración"
    # detectada supera el límite duro, se re-parte de forma mecánica (por comas y, si aún es
    # necesario, por cantidad de palabras) para no generar fragmentos desproporcionados.
    resultado = []
    for o in oraciones:
        resultado.extend(_dividir_oracion_larga(o))
    return [o for o in resultado if o]


LIMITE_DURO_PALABRAS = int(1.5 * TAMANO_OBJETIVO_PALABRAS)


def _dividir_oracion_larga(oracion, limite=LIMITE_DURO_PALABRAS):
    palabras = oracion.split(" ")
    if len(palabras) <= limite:
        return [oracion]
    # Intento 1: partir por comas (listados) manteniendo trozos cercanos al límite.
    partes_coma = oracion.split(", ")
    if len(partes_coma) > 1:
        trozos, actual = [], ""
        for parte in partes_coma:
            candidato = f"{actual}, {parte}" if actual else parte
            if len(candidato.split(" ")) > limite and actual:
                trozos.append(actual)
                actual = parte
            else:
                actual = candidato
        if actual:
            trozos.append(actual)
        if all(len(t.split(" ")) <= limite for t in trozos):
            return trozos
    # Última opción: partir mecánicamente por cantidad de palabras.
    return [" ".join(palabras[i:i + limite]) for i in range(0, len(palabras), limite)]


def construir_chunks(texto, tamano_objetivo=TAMANO_OBJETIVO_PALABRAS, solapamiento=SOLAPAMIENTO_ORACIONES):
    """Agrupa oraciones consecutivas en fragmentos de ~tamano_objetivo palabras, con solapamiento."""
    oraciones = dividir_oraciones(texto)
    if not oraciones:
        return []

    # Tope para el solapamiento: si la(s) oración(es) a repetir ya pesan mucho (ej. viene de un
    # listado largo tipo tabla/nombres), no tiene sentido "solapar" con eso -- infla el siguiente
    # fragmento en vez de darle contexto. Se limita a una fracción chica del tamaño objetivo.
    TOPE_SOLAPAMIENTO_PALABRAS = max(30, tamano_objetivo // 5)

    chunks, actual, palabras_actual = [], [], 0
    for oracion in oraciones:
        n_palabras = len(oracion.split())
        if actual and palabras_actual + n_palabras > tamano_objetivo:
            chunks.append(" ".join(actual))
            candidato_solape = actual[-solapamiento:] if solapamiento else []
            if sum(len(o.split()) for o in candidato_solape) <= TOPE_SOLAPAMIENTO_PALABRAS:
                actual = candidato_solape
            else:
                actual = []
            palabras_actual = sum(len(o.split()) for o in actual)
        actual.append(oracion)
        palabras_actual += n_palabras
    if actual:
        chunks.append(" ".join(actual))
    return chunks


def _slug(texto, largo=40):
    limpio = re.sub(r"[^a-zA-Z0-9]+", "-", texto).strip("-").lower()
    return limpio[:largo] or "sin-nombre"


def extraer_numero_fenomeno(fenomeno_carpeta):
    """'F1_IA_y_Capacidades_Estrategicas' -> '1' (formato exigido por la Tabla 1: 1, 2 o 3)."""
    m = re.match(r"F(\d)", fenomeno_carpeta)
    return m.group(1) if m else fenomeno_carpeta


def procesar_documentos(ruta_entrada, ruta_salida):
    total_docs, total_chunks, sin_doc_id = 0, 0, 0
    with open(ruta_entrada, encoding="utf-8") as f_in, open(ruta_salida, "w", encoding="utf-8") as f_out:
        for linea in f_in:
            doc = json.loads(linea)
            total_docs += 1
            textos_chunk = construir_chunks(doc["texto"])

            doc_id = doc.get("doc_id")
            if not doc_id:
                sin_doc_id += 1
                doc_id = f"SINID-{_slug(doc['fuente'])}"

            for posicion, texto_chunk in enumerate(textos_chunk):
                registro = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}-{posicion:03d}",
                    "fuente": doc["fuente"],
                    "formato": doc["formato"],
                    "fenomeno": extraer_numero_fenomeno(doc["fenomeno"]),
                    "posicion": posicion,
                    "num_tokens": len(texto_chunk.split()),  # aproximado por palabras, ver docstring
                    "texto": texto_chunk,
                }
                f_out.write(json.dumps(registro, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Documentos procesados: {total_docs} ({sin_doc_id} sin doc_id oficial, usando id de respaldo)")
    print(f"Fragmentos generados: {total_chunks}")
    print(f"Promedio de fragmentos por documento: {total_chunks / total_docs:.1f}" if total_docs else "")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python chunking.py <documentos_extraidos.jsonl> [chunks.jsonl]")
        sys.exit(1)
    entrada = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else "chunks.jsonl"
    procesar_documentos(entrada, salida)
