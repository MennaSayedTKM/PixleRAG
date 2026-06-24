"""
search.py
Embed a text query via the Colab server and search the FAISS index.

Pipeline:
  1. FAISS retrieves a large candidate pool (top_k * 10 tiles)
  2. Deduplicate tiles -> unique pages, keeping best score per page
  3. MMR (Max Marginal Relevance) reorders pages to balance
     relevance (similarity to query) vs diversity (dissimilarity to
     already-selected pages) — prevents text-heavy pages from
     monopolising the top results.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from PIL import Image

from embed_client import EmbedClient, EmbedServerError
from config import TOP_K as DEFAULT_TOP_K, MMR_LAMBDA

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "metadata.json"


class SearchResult:
    def __init__(self, vector_id: int, score: float, source: str, page: int,
                 tile_path: str, full_page_path: str):
        self.vector_id = vector_id
        self.score = score          # MMR score (not raw cosine)
        self.source = source
        self.page = page
        self.tile_path = tile_path
        self.full_page_path = full_page_path
        self._image: Optional[Image.Image] = None

    @property
    def image(self) -> Optional[Image.Image]:
        """Always returns the full-page image for display."""
        if self._image is None:
            for path in [self.full_page_path, self.tile_path]:
                if path and Path(path).exists():
                    self._image = Image.open(path).convert("RGB")
                    break
        return self._image

    def __repr__(self):
        return f"SearchResult(source={self.source!r}, page={self.page}, score={self.score:.4f})"


def _mmr(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_scores: list[float],
    top_k: int,
    lam: float = MMR_LAMBDA,
) -> list[int]:
    """
    Max Marginal Relevance selection.

    Args:
        query_vec:        (1, dim) unit-normalised query embedding.
        candidate_vecs:   (n, dim) unit-normalised candidate embeddings.
        candidate_scores: raw cosine scores from FAISS (length n).
        top_k:            number of results to select.
        lam:              relevance/diversity trade-off (0=diversity, 1=relevance).

    Returns:
        List of selected indices into candidate_vecs, in MMR order.
    """
    n = len(candidate_scores)
    top_k = min(top_k, n)
    selected = []
    remaining = list(range(n))

    # Normalise relevance scores to [0, 1]
    scores = np.array(candidate_scores, dtype=np.float32)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        rel = (scores - s_min) / (s_max - s_min)
    else:
        rel = np.ones(n, dtype=np.float32)

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: highest relevance
            best_idx = max(remaining, key=lambda i: rel[i])
        else:
            # Compute max similarity to already-selected vectors
            sel_vecs = candidate_vecs[selected]  # (k, dim)
            # cosine similarity between each remaining candidate and all selected
            sims = candidate_vecs[remaining] @ sel_vecs.T  # (r, k)
            max_sim_to_selected = sims.max(axis=1)         # (r,)

            mmr_scores = (
                lam * rel[remaining]
                - (1 - lam) * max_sim_to_selected
            )
            best_idx = remaining[int(np.argmax(mmr_scores))]

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def search(query: str, client: EmbedClient, top_k: int = 5) -> list[SearchResult]:
    """
    Embed `query`, retrieve candidates from FAISS, deduplicate to unique
    pages, then apply MMR to return a diverse and relevant top-k.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "No FAISS index found. Upload and index some documents first."
        )
    if not META_PATH.exists():
        raise FileNotFoundError("Metadata file missing. Re-index your documents.")

    vec = client.embed_text(query)
    query_vec = np.array(vec, dtype=np.float32).reshape(1, -1)

    index = faiss.read_index(str(INDEX_PATH))

    # Pull a large candidate pool to give MMR enough to work with
    candidates = min(top_k * 10, index.ntotal)
    scores, ids = index.search(query_vec, candidates)

    with open(META_PATH) as f:
        meta = json.load(f)
    meta_by_id = {m["vector_id"]: m for m in meta}

    # Collect candidates
    candidates_list = []
    for score, vid in zip(scores[0], ids[0]):
        if vid == -1:
            continue
        m = meta_by_id.get(int(vid))
        if m is None:
            continue
        candidates_list.append({"score": float(score), "meta": m, "vector_id": int(vid)})

    if not candidates_list:
        return []

    # Retrieve embedding vectors for MMR
    page_vecs = []
    page_scores = []
    for entry in candidates_list:
        try:
            v = index.reconstruct(entry["vector_id"])
        except Exception:
            v = np.zeros(query_vec.shape[1], dtype=np.float32)
        page_vecs.append(v)
        page_scores.append(entry["score"])

    page_vecs_np = np.stack(page_vecs).astype(np.float32)

    # Apply MMR
    mmr_indices = _mmr(query_vec, page_vecs_np, page_scores, top_k=top_k)

    results = []
    for idx in mmr_indices:
        entry = candidates_list[idx]
        m = entry["meta"]
        results.append(
            SearchResult(
                vector_id=entry["vector_id"],
                score=entry["score"],
                source=m["source"],
                page=m["page"],
                tile_path=m["tile_path"],
                full_page_path=m["tile_path"],
            )
        )

    return results
