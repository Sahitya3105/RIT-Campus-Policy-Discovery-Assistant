"""
HyDE Retriever — Multi-Hypothesis HyDE + Hybrid RAG
====================================================
Core retrieval module for the MSRIT AI Campus Policy Discovery Assistant.

Architecture guarantee:
  - HyDE hypotheses are used ONLY for dense-embedding retrieval signal.
  - They are NEVER injected into real_chunks or passed to the LLM synthesis step.
  - hyde_leakage_pct is architecturally guaranteed to be 0.0.
"""

import logging
import math
import os
import pickle
import time

import openai
from dotenv import load_dotenv
load_dotenv()
from fastembed import TextEmbedding
from flashrank import Ranker, RerankRequest
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (overridable via environment variables)
# ---------------------------------------------------------------------------
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "msrit_policies")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "8"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))
BM25_PATH = os.path.join(QDRANT_PATH, "bm25_index.pkl")


# ---------------------------------------------------------------------------
# HyDERetriever
# ---------------------------------------------------------------------------

class HyDERetriever:
    """
    Multi-Hypothesis HyDE + Hybrid RAG retriever.

    Pipeline
    --------
    1. generate_multi_hyde(query)         →  2 synthetic policy excerpts
    2. embed(query) + embed(h1) + embed(h2) →  3 × Qdrant dense searches
    3. bm25_search(query)                 →  1 sparse search
    4. RRF fusion of all 4 result lists
    5. Take top-8 → flashrank reranker → top-3 real chunks
    6. real_chunks contain ONLY actual handbook text; hyde_snippets returned
       separately for UI display / benchmarking ONLY.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        load_dotenv()

        logger.info("Initialising HyDERetriever …")

        # Dense embedder (fastembed)
        self._embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

        # Qdrant (local persistent)
        self._qdrant = QdrantClient(path=QDRANT_PATH)

        # BM25 index (pre-built and serialised by the ingestion pipeline)
        with open(BM25_PATH, "rb") as fh:
            self.bm25, self.bm25_corpus, self.bm25_doc_ids = pickle.load(fh)
        logger.info("BM25 index loaded — %d documents", len(self.bm25_doc_ids))

        # Flashrank reranker
        self._reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        # OpenAI-compatible LLM client (for HyDE generation)
        self._llm = openai.OpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
        )

        logger.info("HyDERetriever ready.")

    # ------------------------------------------------------------------
    # HyDE Generation
    # ------------------------------------------------------------------

    _HYDE_SYSTEM_PROMPT = (
        "You are a formal MSRIT University Policy Compiler. "
        "Your task is to generate hypothetical policy excerpts that MIGHT answer "
        "a student's question, written in the exact bureaucratic language of a "
        "university student handbook. Generate TWO distinct formal policy excerpts "
        "covering DIFFERENT regulatory interpretations. Format as:\n"
        "HYPOTHESIS 1:\n"
        "[formal policy text, 60-80 words, citing specific percentages, deadlines, "
        "and procedures]\n\n"
        "HYPOTHESIS 2:\n"
        "[alternative formal policy text, 60-80 words, covering a different aspect "
        "or clause]"
    )

    def generate_multi_hyde(self, query: str) -> tuple[list[str], float]:
        """
        Call the LLM to produce two hypothetical policy excerpts for *query*.

        Returns
        -------
        (hypotheses, latency_ms)
            hypotheses  — always a 2-element list; empty strings on failure
            latency_ms  — wall-clock time for the LLM call in milliseconds
        """
        t0 = time.perf_counter()
        try:
            user_msg = (
                f'Student query: "{query}"\n\n'
                "Generate 2 hypothetical university policy excerpts that would "
                "answer this."
            )
            model_to_use = os.getenv("LLM_MODEL_NAME", os.getenv("LLM_MODEL", "qwen3.8:27b")).strip()
            response = self._llm.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": self._HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.6,
                max_tokens=650,
            )
            raw = response.choices[0].message.content or ""
            if not raw and getattr(response.choices[0].message, "reasoning", None):
                raw = response.choices[0].message.reasoning
            hypotheses = self._parse_hypotheses(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM HyDE generation unavailable (%s). Applying domain-calibrated bureaucratic expansion fallback.", exc)
            q_lower = query.lower()
            if any(k in q_lower for k in ["band", "gig", "music", "sport", "cultural", "represent", "competition"]):
                hypotheses = [
                    "Ramaiah Institute regulations concerning academic accommodations and attendance condonation under Clause 10.1 for students representing the Institute in cultural activities, music, and sports.",
                    "Laboratory practical session makeup guidelines, compensatory practical evaluations, and internal assessment reschedule for official representation approved by Principal and HOD."
                ]
            elif any(k in q_lower for k in ["hospital", "medic", "sick", "ill", "doctor", "health"]):
                hypotheses = [
                    "Autonomous Institute condonation of attendance shortage up to 10% by the Principal under Clause 10.1 on medical grounds, hospitalization, or acute illness with medical certificate submission.",
                    "Award of transitional grade I under Clause 8.10 for students with valid attendance unable to appear for SEE due to serious medical illness or hospitalization with make-up exam."
                ]
            elif any(k in q_lower for k in ["attend", "shortage", "debar", "dx", "summer", "makeup"]):
                hypotheses = [
                    "Autonomous Institute minimum attendance requirement of 85% under Clause 10.1, condonation floor of 75%, and award of DX grade under Clause 10.3 and Addendum Clause 21.2 barring eligibility for SEE.",
                    "Registration for Fast Track Summer Semester 10-week term under Addendum Clause 21.3 for clearing DX attendance shortage and re-earning CIE credits."
                ]
            else:
                hypotheses = [
                    f"Official academic regulations and evaluation guidelines governing {query}.",
                    f"Academic council autonomous policy handbook clauses regarding {query} and student eligibility."
                ]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return hypotheses, latency_ms

    @staticmethod
    def _parse_hypotheses(raw: str) -> list[str]:
        """
        Extract the two hypothesis blocks from the LLM's response text.
        Gracefully returns empty strings if the expected markers are absent.
        """
        h1, h2 = "", ""
        try:
            if "HYPOTHESIS 1:" in raw and "HYPOTHESIS 2:" in raw:
                after_h1 = raw.split("HYPOTHESIS 1:", 1)[1]
                h1_raw, after_h2 = after_h1.split("HYPOTHESIS 2:", 1)
                h1 = h1_raw.strip()
                h2 = after_h2.strip()
            elif "HYPOTHESIS 2:" in raw:
                h2 = raw.split("HYPOTHESIS 2:", 1)[1].strip()
            else:
                h1 = raw.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse HyDE hypotheses: %s", exc)
        return [h1, h2]

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> list[float]:
        """Return a dense embedding vector for a single string."""
        # fastembed returns a generator of numpy arrays
        embeddings = list(self._embedder.embed([text]))
        return embeddings[0].tolist()

    # ------------------------------------------------------------------
    # Dense search (Qdrant)
    # ------------------------------------------------------------------

    def _qdrant_search(
        self, embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """
        Nearest-neighbour search in Qdrant.

        Returns
        -------
        List of dicts with keys: clause_id, section_title, category, text, score
        """
        if hasattr(self._qdrant, "query_points"):
            response = self._qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        else:
            hits = self._qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=embedding,
                limit=top_k,
                with_payload=True,
            )
        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                {
                    "clause_id": payload.get("clause_id", ""),
                    "section_title": payload.get("section_title", ""),
                    "category": payload.get("category", ""),
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Sparse search (BM25)
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25Okapi search over the pre-built corpus.

        Returns
        -------
        List of dicts with same keys as _qdrant_search output.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        # Pair scores with doc indices and sort descending
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = indexed[:top_k]

        results = []
        for doc_idx, score in top:
            if score <= 0:
                continue
            doc_id = self.bm25_doc_ids[doc_idx]
            corpus_entry = self.bm25_corpus[doc_idx]
            results.append(
                {
                    "clause_id": corpus_entry.get("clause_id", doc_id),
                    "section_title": corpus_entry.get("section_title", ""),
                    "category": corpus_entry.get("category", ""),
                    "text": corpus_entry.get("text", ""),
                    "score": float(score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self, *result_lists: list[dict], k: int = 60
    ) -> list[dict]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.

        RRF score for a document = Σ  1 / (k + rank_i)
        where rank_i is the 1-based position of the document in list i.

        Returns
        -------
        List of dicts sorted by RRF score descending.  Each dict preserves
        the first-seen payload fields and gains a ``rrf_score`` key.
        """
        rrf_scores: dict[str, float] = {}
        doc_store: dict[str, dict] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                cid = doc["clause_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                if cid not in doc_store:
                    doc_store[cid] = doc

        merged = []
        for cid, rrf_score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            entry = dict(doc_store[cid])
            entry["rrf_score"] = rrf_score
            merged.append(entry)

        return merged

    # ------------------------------------------------------------------
    # Raw baseline (no HyDE, no reranking)
    # ------------------------------------------------------------------

    def raw_baseline_search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Simple dense retrieval without HyDE or reranking.
        Used for baseline comparison only.
        """
        t0 = time.perf_counter()
        embedding = self._embed_text(query)
        results = self._qdrant_search(embedding, top_k=top_k)
        logger.debug(
            "raw_baseline_search took %.1f ms", (time.perf_counter() - t0) * 1000
        )
        return results

    # ------------------------------------------------------------------
    # Full HyDE + Hybrid pipeline
    # ------------------------------------------------------------------

    def hyde_hybrid_search(self, query: str) -> dict:
        """
        Execute the full Multi-Hypothesis HyDE + Hybrid RAG pipeline.

        # HYDE ISOLATION GUARANTEE: The hyde_snippets returned by this function
        # are ONLY for UI display and benchmarking comparison. They are NEVER
        # passed to the LLM synthesis prompt. The real_chunks contain ONLY
        # actual handbook text retrieved from Qdrant/BM25.
        # hyde_leakage_pct is architecturally guaranteed to be 0.0.

        Returns
        -------
        {
            "real_chunks"      : list[dict]   — top-3 real handbook chunks
            "hyde_snippets"    : list[str]    — 2 synthetic texts (display only)
            "raw_top3"         : list[dict]   — raw baseline top-3
            "rrf_scores"       : list[float]
            "reranker_scores"  : list[float]
            "top_score"        : float
            "latency_ms"       : {
                "hyde_generation": float,
                "dense_search"   : float,
                "bm25_search"    : float,
                "rrf_fusion"     : float,
                "reranking"      : float,
                "total"          : float,
            }
        }
        """
        pipeline_start = time.perf_counter()
        latency: dict[str, float] = {}

        # ----------------------------------------------------------
        # 1. Generate HyDE hypotheses
        # ----------------------------------------------------------
        hypotheses, hyde_ms = self.generate_multi_hyde(query)
        latency["hyde_generation"] = hyde_ms
        h1, h2 = hypotheses[0], hypotheses[1]

        # ----------------------------------------------------------
        # 2. Dense search: query + h1 + h2  (3 × Qdrant)
        # ----------------------------------------------------------
        t_dense = time.perf_counter()

        query_emb = self._embed_text(query)
        query_results = self._qdrant_search(query_emb, top_k=10)

        h1_results: list[dict] = []
        if h1:
            h1_emb = self._embed_text(h1)
            h1_results = self._qdrant_search(h1_emb, top_k=10)

        h2_results: list[dict] = []
        if h2:
            h2_emb = self._embed_text(h2)
            h2_results = self._qdrant_search(h2_emb, top_k=10)

        latency["dense_search"] = (time.perf_counter() - t_dense) * 1000.0

        # ----------------------------------------------------------
        # 3. BM25 sparse search
        # ----------------------------------------------------------
        t_bm25 = time.perf_counter()
        bm25_results = self._bm25_search(query, top_k=10)
        latency["bm25_search"] = (time.perf_counter() - t_bm25) * 1000.0

        # ----------------------------------------------------------
        # 4. RRF fusion of all 4 result lists
        # ----------------------------------------------------------
        t_rrf = time.perf_counter()
        fused = self._reciprocal_rank_fusion(
            query_results, h1_results, h2_results, bm25_results
        )
        latency["rrf_fusion"] = (time.perf_counter() - t_rrf) * 1000.0

        rrf_scores_all = [doc.get("rrf_score", 0.0) for doc in fused]

        # ----------------------------------------------------------
        # 5. Take top TOP_K_RERANK candidates for reranking
        # ----------------------------------------------------------
        candidates = fused[:TOP_K_RERANK]

        # ----------------------------------------------------------
        # 6. Flashrank reranking → top FINAL_TOP_K
        # ----------------------------------------------------------
        t_rerank = time.perf_counter()

        passages = [
            {"id": i, "text": doc["text"], "meta": doc}
            for i, doc in enumerate(candidates)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = self._reranker.rerank(rerank_request)

        # reranked is a list of passage dicts with an added 'score' key
        reranked_sorted = sorted(reranked, key=lambda x: x.get("score", 0.0), reverse=True)
        top_reranked = reranked_sorted[:FINAL_TOP_K]

        latency["reranking"] = (time.perf_counter() - t_rerank) * 1000.0

        # ----------------------------------------------------------
        # 7. Build real_chunks — ONLY actual handbook text
        #    (HyDE snippets are NEVER included here)
        # ----------------------------------------------------------
        real_chunks: list[dict] = []
        reranker_scores: list[float] = []

        for passage in top_reranked:
            meta = passage.get("meta", {})
            reranker_scores.append(float(passage.get("score", 0.0)))
            real_chunks.append(
                {
                    "clause_id": meta.get("clause_id", ""),
                    "section_title": meta.get("section_title", ""),
                    "category": meta.get("category", ""),
                    "text": meta.get("text", ""),
                    "score": float(passage.get("score", 0.0)),
                    "rrf_score": float(meta.get("rrf_score", 0.0)),
                }
            )

        top_score = reranker_scores[0] if reranker_scores else 0.0

        # ----------------------------------------------------------
        # 8. Raw baseline (for comparison)
        # ----------------------------------------------------------
        raw_top3 = self.raw_baseline_search(query, top_k=FINAL_TOP_K)

        # ----------------------------------------------------------
        # Totals
        # ----------------------------------------------------------
        latency["total"] = (time.perf_counter() - pipeline_start) * 1000.0

        logger.info(
            "hyde_hybrid_search | query=%r | top_score=%.4f | total_ms=%.1f",
            query,
            top_score,
            latency["total"],
        )

        return {
            # HYDE ISOLATION GUARANTEE: The hyde_snippets returned by this
            # function are ONLY for UI display and benchmarking comparison.
            # They are NEVER passed to the LLM synthesis prompt.
            # The real_chunks contain ONLY actual handbook text retrieved
            # from Qdrant/BM25.
            # hyde_leakage_pct is architecturally guaranteed to be 0.0.
            "real_chunks": real_chunks,
            "hyde_snippets": [h1, h2],          # display / audit only
            "raw_top3": raw_top3,
            "rrf_scores": rrf_scores_all[:TOP_K_RERANK],
            "reranker_scores": reranker_scores,
            "top_score": top_score,
            "latency_ms": latency,
        }
