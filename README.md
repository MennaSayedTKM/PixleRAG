# PixleRAG

**PixleRAG** is a visual document retrieval and question-answering system that works entirely from page images — no text extraction, no OCR, no parsing. It embeds document pages as images, retrieves the most relevant ones with FAISS, and uses GPT-4o vision to read and answer questions directly from the visual content.

This makes it effective for documents where text extraction fails or loses meaning: scanned PDFs, charts, figures, tables, invoices, forms, and mixed-layout reports.

---

## When to use it

| Use case | Why PixleRAG works |
|---|---|
| Scanned PDFs | No OCR needed — pages are embedded as images |
| Charts & figures | GPT-4o reads visual data directly |
| Invoices & forms | Preserves layout and structure |
| Mixed-content docs | Text and visuals treated uniformly |
| Any PDF where copy-paste loses meaning | Pixel-level fidelity |

---

## How it works

```
┌──────────────────────────────────────────────────────────┐
│  Google Colab (GPU)                                      │
│  colab_embed_server.ipynb                                │
│  • Loads Qwen3-VL-Embedding-2B (fp16, CUDA)              │
│  • Serves POST /embed_image  and  POST /embed_text       │
│  • Exposed via ngrok HTTPS tunnel                        │
└─────────────────────┬────────────────────────────────────┘
                      │ HTTPS (ngrok)
┌─────────────────────▼────────────────────────────────────┐
│  Local machine                                           │
│                                                          │
│  app.py          Streamlit GUI                           │
│  api.py          FastAPI REST endpoints                  │
│  ingest.py       PDF/image → page tiles → embeddings     │
│  search.py       Query embedding → FAISS + MMR retrieval │
│  answer.py       Rerank → crop figure → GPT-4o answer    │
│  embed_client.py HTTP client for the Colab server        │
│                                                          │
│  data/tiles/         saved tile PNGs                     │
│  data/index.faiss    FAISS flat inner-product index      │
│  data/metadata.json  vector_id → source / page / path   │
└──────────────────────────────────────────────────────────┘
```

**Answer pipeline:**
1. Query is embedded on Colab → FAISS retrieves top candidates
2. MMR diversifies results (avoids duplicate pages)
3. GPT-4o reranker promotes the page most relevant to the question
4. If a figure is referenced, GPT-4o crops and upscales the region
5. GPT-4o reads the final page(s) at high resolution and synthesises the answer

---

## Setup

### Prerequisites

- Python 3.10+
- A Google account with [Google Colab](https://colab.research.google.com) access
- A free [ngrok account](https://ngrok.com) (for the Colab tunnel)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env`:

```env
EMBED_API_URL=https://xxxx-xx-xx-xxx-xxx.ngrok-free.app   # filled after step 3
OPENAI_API_KEY=sk-...
```

### 3. Start the Colab embedding server

1. Open `colab_embed_server.ipynb` in Google Colab
2. Set **Runtime → Change runtime type → T4 GPU**
3. Paste your ngrok authtoken into Cell 2
4. **Run all cells** — the last cell prints:
   ```
   ✅  Embedding server is live at: https://xxxx-xx-xx-xxx-xxx.ngrok-free.app
   ```
5. Copy that URL into `EMBED_API_URL` in your `.env`

> **Note:** The ngrok URL changes every time the Colab session restarts (idle timeout ~90 min, hard limit 12 h on free tier). Re-run the notebook and update the URL when this happens.

### 4. Adjust settings (optional)

Edit `config.yaml` to change behaviour without touching code:

```yaml
reranker: gpt-4o                    # "gpt-4o" or "jina-reranker-v2-base-multimodal"
top_k: 5                            # pages returned per query
mmr_lambda: 0.4                     # diversity vs relevance (0=diverse, 1=relevant)
rerank_top_n: 4                     # pages sent to GPT-4o for final answer
pdf_dpi: 300                        # rendering resolution
answer_model: gpt-4o                # model used for reranking and answering
crop_min_px: 900                    # minimum crop width when zooming into a figure
```

---

## Streamlit GUI

```bash
streamlit run app.py
```

The GUI has three tabs:

**Upload & Index**
- Drag in one or more PDFs or images
- Click **Index uploaded files**
- Each page is rendered, embedded, and stored in `data/`

**Search & Ask**
- Type a natural language question
- Click **Search** to retrieve relevant pages with similarity scores
- Click **Ask** for a full answer synthesised from the top pages
- Page images are displayed alongside the answer

**Manage Index**
- View which files are indexed and how many pages each has
- Clear the entire index to start fresh

---

## REST API

```bash
uvicorn api:app --reload --port 8080
```

Interactive docs available at `http://localhost:8080/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check — also pings the embed server |
| `POST` | `/ingest` | Upload a PDF or image file and index it |
| `GET` | `/index` | List all indexed files and page counts |
| `DELETE` | `/index` | Clear the entire index |
| `POST` | `/search` | Retrieve relevant pages for a query (no answer) |
| `POST` | `/ask` | Full pipeline: retrieve → rerank → answer |

### Example: ask a question

```bash
curl -X POST http://localhost:8080/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total amount on the invoice?", "top_k": 5}'
```

```json
{
  "query": "What is the total amount on the invoice?",
  "answer": "The total amount is $4,320.00, as shown in Image 1 (invoice.pdf, page 1).",
  "retrieved": [
    {"source": "invoice.pdf", "page": 1, "score": 0.9821},
    {"source": "invoice.pdf", "page": 2, "score": 0.7134}
  ]
}
```

### Example: ingest a file

```bash
curl -X POST http://localhost:8080/ingest \
  -F "file=@report.pdf"
```

---

## Project structure

```
app.py                     Streamlit GUI
api.py                     FastAPI REST API
ingest.py                  PDF/image → page tiles → FAISS
search.py                  FAISS + MMR retrieval
answer.py                  Rerank + figure crop + GPT-4o synthesis
embed_client.py            HTTP client for the Colab embedding server
config.py                  Loads config.yaml (no hardcoded defaults)
config.yaml                All tunable settings
colab_embed_server.ipynb   Colab notebook — runs the embedding model on GPU
requirements.txt           Local dependencies (no torch/transformers)
.env.example               Secret keys template
data/
  tiles/                   Page tile PNG files (generated at ingest time)
  index.faiss              FAISS index
  metadata.json            Vector ID → source / page / tile path
```
