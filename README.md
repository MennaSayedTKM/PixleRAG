# PixelRAG

A two-part visual document retrieval system.  
Documents (PDFs, images) are rendered to image tiles and embedded with **Qwen3-VL-Embedding-2B** — no text parsing anywhere.  
Retrieval is powered by FAISS; answers are synthesised by **gpt-4o** vision.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  PART 1 — Google Colab (GPU)                        │
│  colab_embed_server.ipynb                           │
│  • Loads Qwen3-VL-Embedding-2B (fp16, CUDA)         │
│  • FastAPI: POST /embed_image  POST /embed_text     │
│  • Exposed via ngrok HTTPS tunnel                   │
└────────────────────┬────────────────────────────────┘
                     │ HTTP (ngrok)
┌────────────────────▼────────────────────────────────┐
│  PART 2 — Local machine                             │
│  app.py  (Streamlit GUI)                            │
│  ├── ingest.py   render PDF/image → tiles → embed   │
│  ├── search.py   embed query → FAISS search         │
│  ├── answer.py   gpt-4o vision synthesis            │
│  └── embed_client.py  HTTP calls to Colab server   │
│                                                     │
│  data/tiles/     saved tile PNGs                    │
│  data/index.faiss  FAISS flat inner-product index   │
│  data/metadata.json  vector_id → source/page/path  │
└─────────────────────────────────────────────────────┘
```

---

## Setup

### Prerequisites

- Python 3.10+
- A free [ngrok account](https://ngrok.com) (for the Colab tunnel)
- An [OpenAI API key](https://platform.openai.com/api-keys) (for answer synthesis)
- A Google account with access to [Google Colab](https://colab.research.google.com)

### 1. Install local dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and fill in OPENAI_API_KEY
# (EMBED_API_URL is filled in after step 3)
```

### 3. Start the Colab embedding server

1. Open `colab_embed_server.ipynb` in Google Colab.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Paste your ngrok authtoken into Cell 2.
4. **Run all cells** (Runtime → Run all).
5. Wait for the final cell to print:
   ```
   ✅  Embedding server is live at: https://xxxx-xx-xx-xxx-xxx.ngrok-free.app
   ```
6. Copy that URL.

### 4. Start the local app

```bash
# Option A: set EMBED_API_URL before launching
EMBED_API_URL=https://xxxx-xx-xx-xxx-xxx.ngrok-free.app streamlit run app.py

# Option B: launch and paste the URL into the sidebar settings field
streamlit run app.py
```

---

## Usage

1. **Upload & Index tab** — drag in PDFs or images, click "Index uploaded files".  
   Each file is rendered page-by-page to PNG tiles, embedded via the Colab server, and stored in `data/`.

2. **Search tab** — type a question, click Search.  
   The query is embedded on Colab, FAISS finds the closest tiles, and gpt-4o reads those tiles to write the answer.

3. **Manage Index tab** — view stats or clear the index for a fresh start.

---

## Important: URL refresh

The ngrok URL **changes every time the Colab notebook restarts**.  
Causes of restart:
- Colab idle timeout (~90 minutes of inactivity)
- 12-hour session limit (free tier)
- Manual restart or browser close

When this happens:
1. Re-run all cells in the notebook.
2. Copy the new URL from the last cell.
3. Paste it into the Streamlit sidebar (or update `.env` and restart the app).

---

## Project structure

```
colab_embed_server.ipynb   Colab notebook — embedding server
embed_client.py            HTTP client for the Colab server
ingest.py                  PDF/image → tiles → FAISS
search.py                  Query → FAISS → SearchResult list
answer.py                  gpt-4o vision answer synthesis
app.py                     Streamlit GUI
requirements.txt           Local dependencies (no torch/transformers)
.env.example               Secret keys template
data/
  tiles/                   Saved tile PNG files
  index.faiss              FAISS index (created after first ingest)
  metadata.json            Vector ID → source/page/path mapping
```
