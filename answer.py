"""
answer.py
Synthesise an answer from retrieved pages using gpt-4o vision.

Pipeline:
  1. Rerank: send all candidate pages at low resolution to gpt-4o — it
     returns the indices of relevant pages in order of relevance.
     This promotes chart/figure pages that FAISS ranked low.
  2. Crop (figure queries only): if the query references a specific figure,
     crop + upscale the relevant region on the top reranked page.
  3. Synthesise: send top-N reranked pages (+ crop if applicable) at high
     resolution for the final answer.
"""

import base64
import io
import os
import re
from typing import Optional

from PIL import Image

from search import SearchResult
from config import RERANKER, RERANK_TOP_N, CROP_MIN_PX, ANSWER_MODEL

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
JINA_API_KEY   = os.environ.get("JINA_API_KEY", "")
DEFAULT_MODEL  = ANSWER_MODEL


def _image_to_data_url(img: Image.Image, detail: str = "high") -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail},
    }


def _has_figure_reference(query: str) -> bool:
    pattern = r"\b(figure|fig|table|chart|graph|diagram)\s*[\d]+"
    return bool(re.search(pattern, query, re.IGNORECASE))


def _rerank_gpt4o(query: str, results: list[SearchResult], client, model: str) -> list[SearchResult]:
    """Rerank using gpt-4o vision: send thumbnails, ask for relevance order."""
    content = [{
        "type": "text",
        "text": (
            f"I will show you {len(results)} document page images numbered 1 to {len(results)}.\n"
            f"Question: {query}\n\n"
            "For each image, decide if it contains information that helps answer the question.\n"
            "Return ONLY a JSON array of the image numbers that are relevant, ordered from most "
            "to least relevant. Example: [3, 1, 5]\n"
            "If none are relevant return an empty array: []"
        )
    }]
    for i, r in enumerate(results):
        img = r.image
        if img is None:
            continue
        content.append({"type": "text", "text": f"Image {i+1} — {r.source}, page {r.page}:"})
        content.append(_image_to_data_url(img, detail="low"))

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=128,
    )
    raw = resp.choices[0].message.content.strip()
    start, end = raw.find("["), raw.rfind("]")
    ranked_1based = __import__("json").loads(raw[start:end+1]) if start != -1 else []
    ranked_indices = [int(i) - 1 for i in ranked_1based if 1 <= int(i) <= len(results)]

    seen = set(ranked_indices)
    reranked = [results[i] for i in ranked_indices if i < len(results)]
    reranked += [r for i, r in enumerate(results) if i not in seen]
    return reranked


def _rerank_jina(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Rerank using jina-reranker-v2-base-multimodal via Jina AI API."""
    import requests as req_lib

    documents, valid_indices = [], []
    for i, r in enumerate(results):
        img = r.image
        if img is None:
            continue
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        documents.append({"image": f"data:image/png;base64,{b64}"})
        valid_indices.append(i)

    if not documents:
        return results

    resp = req_lib.post(
        "https://api.jina.ai/v1/rerank",
        json={
            "model": "jina-reranker-v2-base-multimodal",
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        },
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()

    scored = sorted(resp.json()["results"], key=lambda x: x["relevance_score"], reverse=True)
    ranked_indices = [valid_indices[item["index"]] for item in scored]

    seen = set(ranked_indices)
    reranked = [results[i] for i in ranked_indices if i < len(results)]
    reranked += [r for i, r in enumerate(results) if i not in seen]
    return reranked


def _rerank(query: str, results: list[SearchResult], client, model: str) -> list[SearchResult]:
    """
    Route to the reranker configured in .env via RERANKER=gpt4o|jina.
    Falls back to gpt4o if the chosen reranker fails or is misconfigured.
    """
    if len(results) <= 1:
        return results

    if RERANKER == "jina-reranker-v2-base-multimodal":
        if not JINA_API_KEY:
            raise ValueError("reranker=jina-reranker-v2-base-multimodal but JINA_API_KEY is not set in .env")
        try:
            return _rerank_jina(query, results)
        except Exception as e:
            print(f"[rerank] Jina failed ({e}), falling back to gpt-4o")
            return _rerank_gpt4o(query, results, client, model)


    # default: gpt4o
    try:
        return _rerank_gpt4o(query, results, client, model)
    except Exception:
        return results  # last resort: original MMR order


def _locate_and_crop(query: str, img: Image.Image, client, model: str) -> Optional[Image.Image]:
    """
    Ask gpt-4o for the bounding box of the relevant figure on this page.
    Returns an upscaled crop, or None if not found / too small.
    Uses detail="high" since we already know this is the right page.
    """
    prompt = (
        f"Question: {query}\n\n"
        "Identify the specific figure, chart, table, or diagram on this page that is most "
        "relevant to the question above.\n"
        "Return ONLY a JSON object with its bounding box as percentages of page dimensions "
        "(0=top/left, 100=bottom/right):\n"
        "{\"top\": <n>, \"left\": <n>, \"bottom\": <n>, \"right\": <n>}\n"
        "If the relevant figure is not on this page, return: {}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    _image_to_data_url(img, detail="high"),
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=64,
        )
        raw = resp.choices[0].message.content.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1:
            return None
        box = __import__("json").loads(raw[start:end + 1])
        if not box or not all(k in box for k in ("top", "left", "bottom", "right")):
            return None
        if (box["bottom"] - box["top"]) < 5 or (box["right"] - box["left"]) < 5:
            return None
    except Exception:
        return None

    w, h = img.size
    pad_x, pad_y = int(0.02 * w), int(0.02 * h)
    left   = max(0, int(box["left"]   / 100 * w) - pad_x)
    top    = max(0, int(box["top"]    / 100 * h) - pad_y)
    right  = min(w, int(box["right"]  / 100 * w) + pad_x)
    bottom = min(h, int(box["bottom"] / 100 * h) + pad_y)

    crop = img.crop((left, top, right, bottom))
    cw, ch = crop.size
    if cw < CROP_MIN_PX:
        scale = CROP_MIN_PX / cw
        crop = crop.resize((int(cw * scale), int(ch * scale)), Image.LANCZOS)
    return crop


def synthesise_answer(
    query: str,
    results: list[SearchResult],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    skip_rerank: bool = False,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required: pip install openai")

    key = api_key or OPENAI_API_KEY
    if not key:
        raise ValueError(
            "OpenAI API key is not set. "
            "Set OPENAI_API_KEY in your environment or .env file."
        )

    client = OpenAI(api_key=key)

    # Step 1 — rerank all retrieved pages to promote the relevant ones
    if skip_rerank or len(results) <= 1:
        reranked = results
    else:
        reranked = _rerank(query, results, client, model)

    top_pages = reranked[:RERANK_TOP_N]

    # Step 2 — for figure queries, crop the relevant region on the top page
    crop = None
    if _has_figure_reference(query) and top_pages:
        top_img = top_pages[0].image
        if top_img is not None:
            crop = _locate_and_crop(query, top_img, client, model)

    # Step 3 — synthesise answer
    content = []

    if crop is not None:
        content.append({
            "type": "text",
            "text": f"[Zoomed crop of relevant figure — {top_pages[0].source}, page {top_pages[0].page}]"
        })
        content.append(_image_to_data_url(crop, detail="high"))
        content.append({
            "type": "text",
            "text": f"[Full page — {top_pages[0].source}, page {top_pages[0].page}]"
        })
        content.append(_image_to_data_url(top_pages[0].image, detail="high"))
        for i, r in enumerate(top_pages[1:]):
            img = r.image
            if img is None:
                continue
            content.append({"type": "text", "text": f"[Image {i + 2}: {r.source}, page {r.page}]"})
            content.append(_image_to_data_url(img, detail="high"))
    else:
        for i, r in enumerate(top_pages):
            img = r.image
            if img is None:
                continue
            content.append({"type": "text", "text": f"[Image {i + 1}: {r.source}, page {r.page}]"})
            content.append(_image_to_data_url(img, detail="high"))

    if not content:
        raise ValueError("None of the retrieved results have loadable tile images.")

    content.append({"type": "text", "text": f"\nQuestion: {query}"})

    system_prompt = (
        "You are a precise document analyst with expertise in reading charts, tables, and figures. "
        "Pay close attention to chart legends, axis labels, line colors, and data point values. "
    )
    if crop is not None:
        system_prompt += (
            "The first image is a zoomed-in crop of the most relevant figure — "
            "use it to read fine details like legend labels, axis values, heatmap cell numbers, and line colors. "
            "The second image is the full page for context. "
        )
    system_prompt += (
        "Answer the user's question using ONLY information visible in the provided images. "
        "Be specific — state exact names, numbers, or values. "
        "Read alphanumeric codes (IDs, invoice numbers) character by character, preserving all hyphens and separators exactly. "
        "After your answer, cite which image(s) you used. "
        "If the answer cannot be found in the images, say so explicitly."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}") from e

    return response.choices[0].message.content
