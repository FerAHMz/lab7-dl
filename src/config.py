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

# Iteraciones de SGNS: cada una cambia un hiperparametro respecto de la base S1
SGNS_BASE = dict(dim=100, window=5, negatives=5, sample=1e-4, min_count=5, lr=0.01,
                 epochs=5, batch_size=32_768, fraction=1.0, seed=SEED)
SGNS_ITERATIONS = {
    "S1": {},                       # base
    "S2": {"lr": 0.003},            # learning rate
    "S3": {"lr": 0.03},
    "S4": {"dim": 50},              # dimension
    "S5": {"dim": 300},
    "S6": {"fraction": 0.25},       # tamano del corpus
    "S7": {"fraction": 0.50},
    "S8": {"window": 2},            # tamano de ventana
    "S9": {"window": 10},
    "S10": {"negatives": 15},       # numero de negativos
    "S11": {"negatives": 2},
    "S12": {"sample": 1e-5},        # umbral de submuestreo
    "S13": {"min_count": 20},       # min_count
}


def sgns_config(name, **extra):
    return {"name": name, **SGNS_BASE, **SGNS_ITERATIONS.get(name, {}), **extra}

# Configuracion final (seccion 4.2): mejores valores de las iteraciones, con d=100 para comparar con GloVe
SGNS_BEST = {"negatives": 15, "epochs": 10}
