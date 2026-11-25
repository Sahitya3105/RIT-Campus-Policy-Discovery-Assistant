"""
ingestion.py — Ingestion pipeline for MSRIT Revised Policy Handbook.

Parses data/msrit_policy_handbook.md containing official MSRIT Autonomous Regulations,
extracting all statutory and addendum clauses across 21 chapters.
Computes dense vector embeddings using fastembed (BAAI/bge-small-en-v1.5),
upserts into local persistent Qdrant (./qdrant_storage),
and serializes a BM25 index (./qdrant_storage/bm25_index.pkl).
"""

import os
import re
import pickle
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
HANDBOOK_PATH = BASE_DIR / "data" / "msrit_policy_handbook.md"
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_storage"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "msrit_policies")
BM25_PATH = Path(QDRANT_PATH) / "bm25_index.pkl"


def parse_handbook(file_path: Path) -> list[dict]:
    """Parse msrit_policy_handbook.md into structured clause dicts."""
    if not file_path.exists():
        raise FileNotFoundError(f"Policy handbook not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by clause starts
    parts = re.split(r"\n(?=\*\s+\*\*(?:Clause|Addendum Clause))", content)

    clauses = []
    current_section = "General Regulations"

    for part in parts:
        # Check if this part contains a section header
        sec_match = re.search(r"##\s+(\d+\.\s+.*?)\r?\n", part)
        if sec_match:
            current_section = sec_match.group(1).strip()

        clause_match = re.search(
            r"^\*\s+\*\*(Clause[s]?\s*[\d\.\s–-]+|Addendum Clause\s*[\d\.\s–-]+)(?:\s*\((.*?)\))?:\*\*\s*([\s\S]*)",
            part.strip(),
            re.MULTILINE
        )

        if clause_match:
            raw_cid = clause_match.group(1).strip()
            title = (clause_match.group(2) or "").strip()
            body = clause_match.group(3).strip()

            # Clean trailing dividers and headers
            body = re.sub(r"\n---.*", "", body).strip()

            # Clean citations like [cite: 12] for better semantic search
            clean_body = re.sub(r"\[cite:\s*\d+(?:,\s*\d+)*\]", "", body).strip()

            # Category slug from section
            category_slug = re.sub(r"^\d+\.\s*", "", current_section).lower()
            category_slug = re.sub(r"[^a-z0-9]+", "_", category_slug).strip("_")

            full_display_title = f"{title}" if title else f"{current_section}"
            full_text = f"[{raw_cid} — {full_display_title}]\nSection: {current_section}\n\n{clean_body}"

            clauses.append({
                "clause_id": raw_cid,
                "section_title": full_display_title,
                "category": category_slug,
                "section": current_section,
                "text": full_text,
                "body": clean_body,
            })

    logger.info("Parsed %d clauses from %s", len(clauses), file_path)
    return clauses


def build_and_save_bm25(clauses: list[dict], output_path: Path):
    """Build BM25 index on tokenized clauses and serialize."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bm25_corpus = []
    bm25_doc_ids = []
    tokenized_corpus = []

    for clause in clauses:
        doc_dict = {
            "clause_id": clause["clause_id"],
            "section_title": clause["section_title"],
            "category": clause["category"],
            "text": clause["text"],
        }
        bm25_corpus.append(doc_dict)
        bm25_doc_ids.append(clause["clause_id"])

        tokens = clause["text"].lower().split()
        tokenized_corpus.append(tokens)

    bm25_model = BM25Okapi(tokenized_corpus)

    with open(output_path, "wb") as f:
        pickle.dump((bm25_model, bm25_corpus, bm25_doc_ids), f)

    logger.info("Saved BM25 index with %d documents to %s", len(bm25_doc_ids), output_path)


def ingest_to_qdrant(clauses: list[dict], qdrant_path: str, collection_name: str):
    """Embed clauses and store into local persistent Qdrant."""
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=qdrant_path)
    logger.info("Initializing fastembed TextEmbedding (BAAI/bge-small-en-v1.5)...")
    embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

    texts_to_embed = [c["text"] for c in clauses]
    logger.info("Computing dense embeddings for %d clauses...", len(texts_to_embed))
    embeddings = list(embedder.embed(texts_to_embed))
    dim = len(embeddings[0])
    logger.info("Embedding dimension: %d", dim)

    # Recreate collection
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    points = []
    for idx, (clause, emb) in enumerate(zip(clauses, embeddings)):
        point = PointStruct(
            id=idx,
            vector=emb.tolist(),
            payload={
                "clause_id": clause["clause_id"],
                "section_title": clause["section_title"],
                "category": clause["category"],
                "section": clause.get("section", ""),
                "text": clause["text"],
            }
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Successfully upserted %d points to Qdrant collection '%s'", len(points), collection_name)


def run_ingestion():
    """Main execution function."""
    logger.info("Starting MSRIT Policy Handbook ingestion...")
    clauses = parse_handbook(HANDBOOK_PATH)

    if not clauses:
        raise ValueError("No clauses found in handbook. Check file path and formatting.")

    # 1. Build and save BM25
    build_and_save_bm25(clauses, BM25_PATH)

    # 2. Ingest into Qdrant
    ingest_to_qdrant(clauses, QDRANT_PATH, QDRANT_COLLECTION)

    print("\n" + "=" * 60)
    print("✅ INGESTION COMPLETE")
    print(f"Total Clauses Ingested : {len(clauses)}")
    print(f"Qdrant Storage Path    : {QDRANT_PATH}")
    print(f"BM25 Index Path        : {BM25_PATH}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_ingestion()
