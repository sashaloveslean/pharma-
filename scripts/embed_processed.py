#!/usr/bin/env python3
"""Embed prepared processed files into the local Chroma store."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = json.dumps(value, ensure_ascii=False)
    return clean


def dissolution_docs(path: Path) -> list[dict]:
    docs = []
    for row in read_jsonl(path):
        text = row.get("text", "")
        metadata = row.get("metadata", {})
        if not text:
            continue
        docs.append(
            {
                "id": row.get("id") or stable_id("dissolution", metadata, text),
                "text": text,
                "metadata": sanitize_metadata({**metadata, "record_type": "dissolution_chunk"}),
            }
        )
    return docs


def chromatography_docs(path: Path) -> list[dict]:
    docs = []
    for row_index, row in enumerate(read_jsonl(path)):
        text = chromatography_text(row)
        docs.append(
            {
                "id": stable_id("chromatography", {**row, "row_index": row_index}, text),
                "text": text,
                "metadata": sanitize_metadata(
                    {
                        "record_type": "chromatography_condition",
                        "source": row.get("source"),
                        "start_page": row.get("start_page"),
                        "section": row.get("section"),
                        "column_phase": row.get("column_phase"),
                        "detector": row.get("detector"),
                        "primary_organic": primary_organic(row),
                        "mobile_phase_ratio": row.get("mobile_phase_ratio"),
                        "flow_ml_min": row.get("flow_ml_min"),
                        "injection_ul": row.get("injection_ul"),
                        "column_temp_c": row.get("column_temp_c"),
                    }
                ),
            }
        )
    return docs


def chromatography_text(row: dict) -> str:
    fields = [
        ("Источник", row.get("source")),
        ("Страница", row.get("start_page")),
        ("Раздел", row.get("section")),
        ("Колонка", row.get("column_raw")),
        ("Фаза колонки", row.get("column_phase")),
        ("Длина колонки, мм", row.get("column_length_mm")),
        ("ID колонки, мм", row.get("column_id_mm")),
        ("Частицы, мкм", row.get("particle_um")),
        ("Температура колонки, C", row.get("column_temp_c")),
        ("Детектор", row.get("detector")),
        ("Длины волн, нм", row.get("wavelengths_nm")),
        ("Подвижная фаза", row.get("mobile_phase_raw")),
        ("pH", row.get("mobile_phase_ph")),
        ("Растворители", row.get("solvents")),
        ("Расход, мл/мин", row.get("flow_ml_min")),
        ("Инжекция, мкл", row.get("injection_ul")),
        ("Время анализа, мин", row.get("runtime_min")),
        ("Реагенты", row.get("reagents")),
        ("Соотношение фаз", row.get("mobile_phase_ratio")),
        ("Компоненты фаз", row.get("mobile_phase_components")),
        ("Режим элюирования", row.get("elution_mode")),
        ("Градиент", row.get("gradient_steps")),
    ]
    return "\n".join(f"{name}: {format_value(value)}" for name, value in fields if value not in (None, [], ""))


def format_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def primary_organic(row: dict) -> str:
    solvents = row.get("solvents") or []
    for solvent in solvents:
        if solvent and solvent != "water":
            return str(solvent)
    return ""


def stable_id(prefix: str, metadata: dict, text: str) -> str:
    key = json.dumps(metadata, ensure_ascii=False, sort_keys=True) + "\n" + text
    return f"{prefix}:{hashlib.sha1(key.encode()).hexdigest()}"


def embed(docs: list[dict], persist_dir: Path, collection_name: str, model: str, reset: bool) -> None:
    import chromadb
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    chroma = chromadb.PersistentClient(path=str(persist_dir))
    if reset:
        try:
            chroma.delete_collection(collection_name)
        except Exception:
            pass
    collection = chroma.get_or_create_collection(collection_name)

    client = OpenAI()
    for start in range(0, len(docs), 64):
        batch = docs[start : start + 64]
        response = client.embeddings.create(model=model, input=[doc["text"] for doc in batch])
        collection.upsert(
            ids=[doc["id"] for doc in batch],
            documents=[doc["text"] for doc in batch],
            metadatas=[doc["metadata"] for doc in batch],
            embeddings=[item.embedding for item in response.data],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--persist-dir", default="vector_store")
    parser.add_argument("--collection", default="molecule_dissolution")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    docs = []
    docs.extend(dissolution_docs(processed_dir / "dissolution_chunks.jsonl"))
    docs.extend(chromatography_docs(processed_dir / "chromatography_conditions.jsonl"))
    if not docs:
        raise SystemExit(f"No processed docs found in {processed_dir}")

    embed(
        docs=docs,
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        model=args.embedding_model,
        reset=args.reset,
    )
    print(f"Embedded {len(docs)} processed docs into {args.collection}")


if __name__ == "__main__":
    main()
