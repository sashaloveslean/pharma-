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

## 6. Predict chromatography conditions for a new molecule

A predictive scaffold turns the extracted НД text into structured HPLC method
labels and predicts likely conditions for an unseen molecule from its structure.

Pipeline:

```text
dissolution_chunks.jsonl
  -> scripts/extract_conditions.py   structured condition blocks (regex, no API key)
  -> scripts/build_dataset.py        one labelled row per molecule + RDKit features
  -> scripts/train_model.py          nearest-analog model + leave-one-out report
  -> scripts/predict_conditions.py   conditions for a new SMILES / INN
```

Run it:

```bash
python3 scripts/extract_conditions.py     # -> data/processed/chromatography_conditions.jsonl
python3 scripts/build_dataset.py          # -> data/processed/labels.csv + dataset.json
python3 scripts/train_model.py            # -> models/nearest_analog.json (+ LOO metrics)
python3 scripts/predict_conditions.py --smiles "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
python3 scripts/predict_conditions.py --inn ibuprofen
```

Predicted output, most actionable first:

- **Reagents** — the chemicals the method needs, grouped by role: organic
  solvents, buffer salts, acids / bases for pH, ion-pairing agents. This is the
  primary output — the reagent shopping list for the method.
- **Conditions** — stationary phase, column length / ID / particle size, column
  temperature, detector + wavelength, organic modifier, mobile phase + pH, flow
  rate, injection volume.

### How it works

- **Labels (y):** `extract_conditions.py` parses the "Хроматографические условия"
  blocks; `build_dataset.py` selects, per document, the single **most complete
  real method** (rather than blending blocks, which produced non-physical values
  and lost the mobile-phase recipe) and back-fills any missing field from the
  document's other blocks.
- **Features (X):** RDKit physicochemical descriptors + a Morgan fingerprint,
  computed from each molecule's SMILES.
- **Model:** with only a few dozen labelled molecules, a trained regressor would
  overfit, so the baseline is *nearest-analog transfer* — it returns the complete,
  coherent method of the structurally closest known molecule (Tanimoto similarity
  of Morgan fingerprints). A field the closest analog lacks is back-filled only
  from other analogs above a similarity threshold, so a molecule-specific recipe
  is never copied from a distant match; otherwise the field is left blank. The
  class in `scripts/model.py` is model-shaped (`fit`/`predict`/`save`/`load`) so a
  learned estimator can replace it once more labelled data exists.

### To improve accuracy

1. **Fill `data/molecule_map.csv`.** Each row maps a source document to its INN
   and **SMILES**. Only rows with SMILES enter the model. Add the missing SMILES,
   verify the pre-filled ones, and flip `verified` to `true`. More molecules =
   better predictions. Rows marked `EXCLUDE` (combinations, allergens,
   homeopathic) are not single small molecules and should stay unmapped.
2. Re-run `build_dataset.py` and `train_model.py`; check the leave-one-out report.

> The prediction is a **method starting point** by analogy, not a validated
> analytical procedure.

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
