#!/usr/bin/env python3
"""Query the local Chroma RAG store."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap


def embed_query(text: str, model: str) -> list[float]:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def retrieve(
    question: str,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    top_k: int,
    candidate_k: int,
) -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)
    query_embedding = embed_query(question, embedding_model)
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=max(top_k, candidate_k),
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict] = []
    for index, document in enumerate(result["documents"][0]):
        metadata = result["metadatas"][0][index]
        distance = result["distances"][0][index]
        keyword_score = score_keywords(question, document, metadata)
        rows.append(
            {
                "text": document,
                "metadata": metadata,
                "distance": distance,
                "keyword_score": keyword_score,
                "hybrid_score": distance - keyword_score,
            }
        )
    return sorted(rows, key=lambda row: row["hybrid_score"])[:top_k]


def score_keywords(question: str, text: str, metadata: dict) -> float:
    haystack = f"{metadata.get('section') or ''}\n{text}".lower()
    question_lower = question.lower()
    score = 0.0

    question_tokens = {
        token
        for token in question_lower.replace("?", " ").replace(",", " ").split()
        if len(token) >= 4
    }
    for token in question_tokens:
        if token in haystack:
            score += 0.015

    if "раствор" in question_lower or "сред" in question_lower:
        phrase_weights = {
            "среда растворения": 0.22,
            "методика растворения": 0.16,
            "объем среды": 0.12,
            "температура среды": 0.12,
            "скорость вращения": 0.10,
            "лопастная мешалка": 0.10,
            "аппарат": 0.06,
            "время отбора": 0.10,
            "0,01 м раствор хлористоводородной кислоты": 0.10,
            "900 мл": 0.10,
            "37": 0.04,
            "75 об/мин": 0.10,
            "10 мл": 0.05,
            "15 мин": 0.05,
        }
        for phrase, weight in phrase_weights.items():
            if phrase in haystack:
                score += weight

    if "сред" in question_lower and "подвижная фаза" in haystack and "среда растворения" not in haystack:
        score -= 0.12

    if "растворение" in (metadata.get("terms") or []):
        score += 0.04
    if metadata.get("section") and "раствор" in str(metadata["section"]).lower():
        score += 0.04

    return min(score, 0.6)


def answer_question(question: str, rows: list[dict], model: str) -> str:
    from openai import OpenAI

    context_blocks = []
    for index, row in enumerate(rows, start=1):
        metadata = row["metadata"]
        page = page_label(metadata)
        context_blocks.append(
            f"[{index}] source={metadata.get('source')} page={page}\n{row['text']}"
        )
    context = "\n\n".join(context_blocks)

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer in Russian. Use the provided context as experimental analogs "
                    "for predictive chromatography and dissolution recommendations. "
                    "Clearly separate source facts from predicted recommendations for the "
                    "new molecule. If the context does not directly contain the requested "
                    "molecule, say that the recommendation is an extrapolation from analogs, "
                    "not a confirmed method. Cite source facts as [1], [2], etc."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext:\n{context}",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""


def page_label(metadata: dict) -> str:
    start = metadata.get("start_page")
    end = metadata.get("end_page")
    if start and end and start != end:
        return f"{start}-{end}"
    if start:
        return str(start)
    return "unknown"


def print_rows(rows: list[dict], max_chars: int) -> None:
    for index, row in enumerate(rows, start=1):
        metadata = row["metadata"]
        print(f"\n[{index}] distance={row['distance']:.4f}")
        print(f"hybrid_score={row['hybrid_score']:.4f} keyword_score={row['keyword_score']:.4f}")
        print(f"source: {metadata.get('source')}")
        print(f"pages: {page_label(metadata)}")
        print(f"section: {metadata.get('section')}")
        print(f"terms: {format_terms(metadata.get('terms'))}")
        print(make_snippet(row["text"], width=max_chars))


def make_snippet(text: str, width: int) -> str:
    flat_text = text.replace("\n", " ")
    lower = flat_text.lower()
    anchors = [
        "среда растворения",
        "методика растворения",
        "аппарат",
        "растворение",
        "подвижная фаза",
        "испытуемый раствор",
        "стандартный раствор",
    ]
    center = None
    for anchor in anchors:
        position = lower.find(anchor)
        if position >= 0:
            center = position
            break
    if center is None:
        return textwrap.shorten(flat_text, width=width)

    start = max(0, center - width // 4)
    snippet = flat_text[start : start + width]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if start > 0:
        snippet = "..." + snippet
    if start + width < len(flat_text):
        snippet = snippet + "..."
    return snippet


def format_terms(terms: object) -> str:
    if not terms:
        return ""
    if isinstance(terms, str):
        try:
            parsed = json.loads(terms)
        except json.JSONDecodeError:
            return terms
        if isinstance(parsed, list):
            return ", ".join(str(term) for term in parsed)
        return str(parsed)
    if isinstance(terms, list):
        return ", ".join(str(term) for term in terms)
    return str(terms)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--persist-dir", default="vector_store")
    parser.add_argument("--collection", default="molecule_dissolution")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--answer-model", default="gpt-4.1-mini")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=15)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()

    rows = retrieve(
        question=args.question,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    print_rows(rows, max_chars=args.max_chars)

    if args.answer:
        print("\nAnswer:")
        print(answer_question(args.question, rows, model=args.answer_model))


if __name__ == "__main__":
    main()
