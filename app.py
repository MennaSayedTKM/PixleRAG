"""
app.py
PixelRAG — Streamlit GUI
"""

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from PIL import Image

load_dotenv()

st.set_page_config(
    page_title="PixelRAG",
    page_icon="🔍",
    layout="wide",
)

EMBED_API_URL = os.environ.get("EMBED_API_URL", "").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

QA_LOG_PATH = Path(__file__).parent / "data" / "qa_log.json"


def _append_qa(question: str, answer: str, results: list):
    QA_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(QA_LOG_PATH.read_text()) if QA_LOG_PATH.exists() else []
    log.append({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "retrieved": [
            {"source": r.source, "page": r.page, "score": round(float(r.score), 4)}
            for r in results
        ],
    })
    QA_LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------
# Global CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
/* ── Font & base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default header decoration ── */
#MainMenu, footer { visibility: hidden; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 28px;
    color: white;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin: 0 0 6px; }
.hero p  { font-size: 1rem; opacity: .85; margin: 0; }

/* ── Tab strip ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #f8f8ff;
    border-radius: 12px;
    padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    color: #6b7280;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #6366f1 !important;
    box-shadow: 0 1px 6px rgba(99,102,241,.15);
}

/* ── Cards ── */
.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
}

/* ── Answer box ── */
.answer-box {
    background: linear-gradient(135deg, #f0f4ff 0%, #f5f0ff 100%);
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 16px 0 24px;
    font-size: 1.02rem;
    line-height: 1.7;
    color: #1e1b4b;
}

/* ── Result tile card ── */
.tile-card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 8px;
    transition: box-shadow .2s;
}
.tile-card:hover { box-shadow: 0 4px 16px rgba(99,102,241,.12); }

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    background: linear-gradient(90deg, #6366f1, #a855f7);
    color: white;
    font-size: .75rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 2px 10px;
    margin-top: 6px;
}

/* ── Source label ── */
.source-label {
    font-size: .82rem;
    color: #6b7280;
    margin-top: 4px;
}

/* ── Status pill ── */
.pill-ok  { background:#d1fae5; color:#065f46; border-radius:999px; padding:3px 12px; font-size:.8rem; font-weight:600; }
.pill-err { background:#fee2e2; color:#991b1b; border-radius:999px; padding:3px 12px; font-size:.8rem; font-weight:600; }

/* ── Indexed file row ── */
.file-row {
    display:flex; align-items:center; gap:10px;
    background:#f9fafb; border-radius:8px;
    padding:10px 14px; margin-bottom:6px;
    font-size:.9rem;
}
.file-row .fname { font-weight:600; color:#111827; flex:1; }
.file-row .fpages { color:#6b7280; font-size:.8rem; }

/* ── Primary button override ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6);
    border: none; border-radius: 8px;
    color: white; font-weight: 600;
    padding: 10px 28px;
    transition: opacity .2s;
}
div.stButton > button[kind="primary"]:hover { opacity: .88; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🔍 PixelRAG</h1>
  <p>Visual document retrieval — no text extraction, pure image understanding</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Helper: get embed client or show error
# ------------------------------------------------------------------
def get_client():
    from embed_client import EmbedClient, EmbedServerError
    if not EMBED_API_URL:
        st.error(
            "⚠️ EMBED_API_URL is not set in your .env file. "
            "Start your Colab notebook, copy the ngrok URL, and add it to .env."
        )
        return None
    try:
        return EmbedClient(EMBED_API_URL)
    except EmbedServerError as e:
        st.error(f"⚠️ Embedding server error: {e}")
        return None


# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_ingest, tab_search, tab_manage = st.tabs(["📥  Upload & Index", "🔍  Search", "🗂  Manage Index"])

# ==================================================================
# TAB 1 — Upload & Index
# ==================================================================
with tab_ingest:
    st.markdown("#### Upload documents")
    st.caption("PDF, PNG, JPG, JPEG, WEBP — rendered to image tiles, no text parsing.")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("⚡ Index uploaded files", type="primary"):
        client = get_client()
        if client:
            from ingest import ingest_file
            from embed_client import EmbedServerError

            for uf in uploaded_files:
                with st.container():
                    st.markdown(f'<div class="card">', unsafe_allow_html=True)
                    st.markdown(f"**📄 {uf.name}**")
                    progress_box = st.empty()
                    log_lines = []

                    def _cb(msg, _box=progress_box, _lines=log_lines):
                        _lines.append(msg)
                        _box.code("\n".join(_lines), language=None)

                    suffix = Path(uf.name).suffix
                    tmp_dir = Path(tempfile.mkdtemp())
                    tmp_path = tmp_dir / uf.name  # preserve original filename
                    tmp_path.write_bytes(uf.read())

                    try:
                        n = ingest_file(tmp_path, client, progress_cb=_cb)
                        st.success(f"✅ Indexed **{n}** tile(s)")
                    except EmbedServerError as e:
                        st.error(
                            f"❌ Embedding server unreachable. "
                            f"Check your Colab notebook is running and the URL is current.\n\n`{e}`"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
                    finally:
                        tmp_path.unlink(missing_ok=True)
                        tmp_dir.rmdir()
                    st.markdown("</div>", unsafe_allow_html=True)

    # Currently indexed
    st.markdown("---")
    st.markdown("#### Currently in the index")
    try:
        from ingest import list_indexed_files
        indexed = list_indexed_files()
        if indexed:
            for item in indexed:
                st.markdown(
                    f'<div class="file-row">'
                    f'<span class="fname">📄 {item["source"]}</span>'
                    f'<span class="fpages">{item["pages"]} tile(s)</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents indexed yet. Upload some files above.")
    except Exception as e:
        st.warning(f"Could not load index list: {e}")


# ==================================================================
# TAB 2 — Search
# ==================================================================
with tab_search:
    st.markdown("#### Ask a question about your documents")

    col_q, col_k = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "Query",
            placeholder="What does the revenue chart on page 3 show?",
            label_visibility="collapsed",
        )
    with col_k:
        top_k = st.number_input("Top-K", min_value=1, max_value=10, value=4, step=1)

    search_btn = st.button("🔍 Search", type="primary", disabled=not query.strip())

    if search_btn and query.strip():
        client = get_client()
        if client:
            from search import search
            from answer import synthesise_answer
            from embed_client import EmbedServerError

            with st.spinner("Searching…"):
                try:
                    results = search(query, client, top_k=top_k)
                except FileNotFoundError as e:
                    st.error(f"⚠️ {e}")
                    results = []
                except EmbedServerError as e:
                    st.error(
                        f"❌ Embedding server unreachable. Check your Colab notebook.\n\n`{e}`"
                    )
                    results = []

            if results:
                # ── Answer ──
                st.markdown("#### 📝 Answer")
                with st.spinner("Asking gpt-4o…"):
                    try:
                        answer = synthesise_answer(
                            query,
                            results,
                            api_key=OPENAI_API_KEY or None,
                        )
                        st.markdown(
                            f'<div class="answer-box">{answer}</div>',
                            unsafe_allow_html=True,
                        )
                        _append_qa(query, answer, results)
                    except ValueError as e:
                        st.error(f"⚠️ {e}")
                    except RuntimeError as e:
                        st.error(f"❌ OpenAI error: {e}")

                # ── Retrieved tiles ──
                st.markdown("#### 🖼 Retrieved Pages")
                cols = st.columns(min(len(results), 4), gap="medium")
                for i, r in enumerate(results):
                    with cols[i % min(len(results), 4)]:
                        st.markdown('<div class="tile-card">', unsafe_allow_html=True)
                        img = r.image
                        if img:
                            # Display at full resolution — no thumbnail downscale
                            buf = io.BytesIO()
                            img.save(buf, format="PNG", optimize=False)
                            buf.seek(0)
                            st.image(buf, use_container_width=True)
                        else:
                            st.warning("Tile image not found.")
                        st.markdown(
                            f'<div class="source-label">📄 {r.source} · p{r.page}</div>'
                            f'<span class="score-badge">score {r.score:.3f}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)


# ==================================================================
# TAB 3 — Manage Index
# ==================================================================
with tab_manage:
    st.markdown("#### Index statistics")

    try:
        from ingest import META_PATH
        if META_PATH.exists():
            with open(META_PATH) as f:
                meta = json.load(f)
            c1, c2 = st.columns(2)
            c1.metric("Total tiles", len(meta))
            c2.metric("Unique files", len({m["source"] for m in meta}))
        else:
            st.info("No index exists yet.")
    except Exception as e:
        st.warning(f"Could not read index stats: {e}")

    st.markdown("---")
    st.markdown("#### Danger zone")
    st.warning(
        "Clearing the index removes all FAISS vectors, metadata, and saved tile images. "
        "You will need to re-index your documents."
    )
    if st.button("🗑 Clear entire index"):
        try:
            from ingest import clear_index
            clear_index()
            st.success("Index cleared.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear index: {e}")
