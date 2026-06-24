"""
config.py
Loads config.yaml once and exposes settings as a simple namespace.
All other modules import from here instead of hardcoding values.
"""

from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

# Reranker
def _require(key: str):
    if key not in _cfg:
        raise KeyError(f"Missing required key '{key}' in config.yaml")
    return _cfg[key]

# Reranker
RERANKER       = str(_require("reranker")).strip()

# Retrieval
TOP_K          = int(_require("top_k"))
MMR_LAMBDA     = float(_require("mmr_lambda"))
RERANK_TOP_N   = int(_require("rerank_top_n"))

# Ingestion
PDF_DPI        = int(_require("pdf_dpi"))
MAX_TILE_PX    = int(_require("max_tile_px"))

# Answer synthesis
ANSWER_MODEL   = str(_require("answer_model"))
CROP_MIN_PX    = int(_require("crop_min_px"))
