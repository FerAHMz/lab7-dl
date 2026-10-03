"""Rutas, semillas e hiperparametros compartidos por todo el laboratorio."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
VECTORS_DIR = RESULTS_DIR / "vectors"
FIG_DIR = ROOT / "figures"

# Caches de Hugging Face y gensim dentro de data/raw (no versionado)
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
os.environ.setdefault("GENSIM_DATA_DIR", str(DATA_DIR / "gensim"))

SEED = 42

# Corpus: prefijo de articulos completos de WikiText-103 (train) hasta superar este numero de tokens
SUBSET_TOKENS = 25_000_000

# Benchmarks y evaluacion
ANALOGY_RESTRICT = 30_000        # top-k palabras mas frecuentes para analogias (convencion de gensim)
SEMANTIC_SECTIONS = 5            # las 5 primeras categorias de questions-words.txt son semanticas
CONTROL_WORDS = ["king", "france", "computer", "good", "january", "run"]

# AG News
AG_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
VAL_FRACTION = 0.10

for d in (DATA_DIR, PROC_DIR, RUNS_DIR, CKPT_DIR, VECTORS_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)
