## base_vectorial/

El índice FAISS, la metadata y el grafo de conocimiento pesan varios GB, así que viven en Drive en vez de GitHub.

👉 **[Descargar carpeta completa](https://drive.google.com/drive/folders/1lUBTt8KhLatsosMQesECipveT45cvkA9?usp=sharing)**

Contiene:
- `grafo/` → grafo.graphml y checkpoint.pkl
- `encoder_multilingual-e5-base/` → index.faiss, metadata.jsonl

Descarga y coloca ambas carpetas aquí mismo antes de correr `generador.py`.

> Descarga también `checkpoint.pkl`, no solo `grafo.graphml`. El sistema busca primero el `.pkl` (carga en segundos); si solo tienes el `.graphml`, carga igual pero tarda varios minutos por ser un archivo XML de 2.3 GB.
