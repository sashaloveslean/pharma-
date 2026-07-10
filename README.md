# RAG Pharma Dissolution Prep

Pipeline for extracting molecule dissolution / solubility / chromatographic solution
information from source documents and preparing it for embeddings.

## 1. Put documents here

Copy PDF, DOCX, TXT, or MD files into:

```bash
data/raw
```

## 2. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## 3. Extract chunks without embeddings

```bash
.venv/bin/python scripts/prepare_rag.py
```

The result will be:

```text
data/processed/dissolution_chunks.jsonl
```

Each line contains:

- `id`
- `text`
- `metadata.source`
- `metadata.start_page` / `metadata.end_page`
- `metadata.section`
- matched `metadata.terms`

## 4. Create embeddings in local Chroma

```bash
export OPENAI_API_KEY="..."
.venv/bin/python scripts/prepare_rag.py --embed
```

By default this writes to:

```text
vector_store
```

Collection name:

```text
molecule_dissolution
```

Default embedding model:

```text
text-embedding-3-small
```

## 5. Query the RAG store

Retrieve the most relevant chunks:

```bash
export OPENAI_API_KEY="..."
.venv/bin/python scripts/query_rag.py "Какая среда растворения используется?"
```

Retrieve chunks and generate an answer from them:

```bash
.venv/bin/python scripts/query_rag.py "Какая среда растворения используется?" --answer
```

## Notes

The extractor searches for Russian and English terms around dissolution and
chromatographic preparation, including:

- раствор, растворение, растворимость, растворитель
- solubility, dissolution, dissolve, solvent, diluent
- mobile phase, sample solution, standard solution, stock solution

If the source document has a very specific heading structure, tune
`DISSOLUTION_TERMS` and `HEADING_RE` in `scripts/prepare_rag.py`.

## Scanned PDFs

If a PDF is a scan, normal PDF text extraction can return almost no text.
The script then automatically falls back to OCR.

For Russian scanned documents, install Tesseract and Russian language data:

```bash
brew install tesseract tesseract-lang
```

Then run:

```bash
.venv/bin/python scripts/prepare_rag.py --ocr-engine tesseract --ocr-dpi 180
```

Higher `--ocr-dpi` values can improve recognition, but make processing slower.
