"""Evaluacion intrinseca de embeddings: analogias (3CosAdd / 3CosMul), similitud y vecinos.

Todos los modelos (SGNS propio, Word2Vec de gensim y GloVe) se manejan como KeyedVectors de
gensim con las palabras ordenadas por frecuencia descendente, para usar la misma API.
"""
from collections import OrderedDict

import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.test.utils import datapath

from .config import ANALOGY_RESTRICT, CONTROL_WORDS, SEMANTIC_SECTIONS

ANALOGIES = datapath("questions-words.txt")
WORDSIM = datapath("wordsim353.tsv")
SIMLEX = datapath("simlex999.txt")


def to_keyed_vectors(words, vectors):
    kv = KeyedVectors(vector_size=vectors.shape[1], count=0, dtype=np.float32)
    kv.add_vectors([str(w) for w in words], np.asarray(vectors, dtype=np.float32))
    return kv


def load_analogies(path=ANALOGIES):
    """Lista de (categoria, a, b, c, d) en minusculas: «a es a b como c es a d»."""
    sections, out = [], []
    for line in open(path, encoding="utf8"):
        if line.startswith(":"):
            sections.append(line[1:].strip())
            continue
        a, b, c, d = line.lower().split()
        out.append((sections[-1], a, b, c, d))
    return sections, out


SECTIONS, QUESTIONS = load_analogies()
SEMANTIC = set(SECTIONS[:SEMANTIC_SECTIONS])


def _unit_matrix(kv, words, device):
    m = torch.tensor(np.stack([kv[w] for w in words]), device=device)
    return m / m.norm(dim=1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def evaluate_analogies(kv, vocab=None, restrict=ANALOGY_RESTRICT, methods=("add", "mul"),
                       device="cpu", batch=2048):
    """Accuracy de analogias por categoria con 3CosAdd y/o 3CosMul.

    Igual que gensim.evaluate_word_analogies: se buscan respuestas solo entre las `restrict`
    palabras mas frecuentes (o en `vocab`, si se pasa un vocabulario compartido), se omiten las
    preguntas con alguna palabra fuera de ese vocabulario y se excluyen a, b y c del resultado.
    """
    words = list(vocab) if vocab is not None else kv.index_to_key[:restrict]
    idx = {w: i for i, w in enumerate(words)}
    qs = [q for q in QUESTIONS if all(w in idx for w in q[1:])]
    M = _unit_matrix(kv, words, device)
    abcd = torch.tensor([[idx[w] for w in q[1:]] for q in qs], device=device).reshape(-1, 4)
    correct = {m: np.zeros(len(qs), dtype=bool) for m in methods}
    for s in range(0, len(qs), batch):
        a, b, c, d = abcd[s:s + batch].T
        sa, sb, sc = M[a] @ M.T, M[b] @ M.T, M[c] @ M.T
        rows = torch.arange(len(a), device=device)
        for m in methods:
            if m == "add":
                score = sb - sa + sc
            else:  # 3CosMul (Levy & Goldberg, 2014), cosenos llevados a [0, 1]
                score = ((sb + 1) / 2) * ((sc + 1) / 2) / ((sa + 1) / 2 + 1e-3)
            for col in (a, b, c):
                score[rows, col] = -np.inf
            correct[m][s:s + len(a)] = (score.argmax(1) == d).cpu().numpy()
    cats = np.array([q[0] for q in qs])
    out = {"n_eval": len(qs), "n_total": len(QUESTIONS), "coverage": len(qs) / len(QUESTIONS)}
    for m in methods:
        ok = correct[m]
        sem = np.isin(cats, list(SEMANTIC))
        out[m] = {
            "semantic": ok[sem].mean() if sem.any() else np.nan,
            "syntactic": ok[~sem].mean() if (~sem).any() else np.nan,
            "total": ok.mean() if len(ok) else np.nan,
            "by_category": OrderedDict((s, (ok[cats == s].mean() if (cats == s).any() else np.nan,
                                            int((cats == s).sum()))) for s in SECTIONS),
        }
    return out


def word_pairs(kv, path):
    """Correlacion de Spearman con juicios humanos (WordSim-353 / SimLex-999) y % de pares OOV."""
    pearson, spearman, oov = kv.evaluate_word_pairs(path, restrict_vocab=len(kv), case_insensitive=True)
    return {"spearman": float(spearman[0]), "pearson": float(pearson[0]), "oov_%": float(oov)}


def neighbors(kv, words=CONTROL_WORDS, k=5):
    return {w: [n for n, _ in kv.most_similar(w, topn=k)] if w in kv.key_to_index else [] for w in words}


def quick_eval(kv, device="cpu"):
    """Metricas que se registran al final de cada epoch."""
    an = evaluate_analogies(kv, methods=("add",), device=device)["add"]
    ws = word_pairs(kv, WORDSIM)
    return {"an_sem": an["semantic"], "an_syn": an["syntactic"], "an_total": an["total"],
            "wordsim": ws["spearman"], "neighbors": neighbors(kv)}


# ----------------------------------------------------------------------------- analogias individuales
def _unit(kv):
    return kv.get_normed_vectors()


def analogia(kv, a, b, c, k=5, exclude=True):
    """Resuelve «a es a b como c es a ?» con 3CosAdd sobre vectores normalizados.

    target = b_hat - a_hat + c_hat; devuelve las k palabras con mayor coseno contra target
    (excluyendo a, b y c si exclude=True) junto con su similitud.
    """
    U = _unit(kv)
    i = kv.key_to_index
    target = U[i[b]] - U[i[a]] + U[i[c]]
    target /= np.linalg.norm(target)
    sims = U @ target
    if exclude:
        sims[[i[a], i[b], i[c]]] = -np.inf
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(kv.index_to_key[j], float(sims[j])) for j in top]


def answer_rank(kv, a, b, c, d, exclude=True):
    """Rango (1 = primero) de la respuesta correcta d y su coseno con el vector resultante."""
    U = _unit(kv)
    i = kv.key_to_index
    target = U[i[b]] - U[i[a]] + U[i[c]]
    target /= np.linalg.norm(target)
    sims = U @ target
    if exclude:
        sims[[i[a], i[b], i[c]]] = -np.inf
    rank = int((sims > sims[i[d]]).sum()) + 1
    return rank, float(sims[i[d]])


def offset_parallelism(kv, pairs):
    """Coseno promedio entre todos los pares de vectores diferencia (b - a) de una relacion."""
    pairs = [(a, b) for a, b in pairs if a in kv.key_to_index and b in kv.key_to_index]
    U = _unit(kv)
    i = kv.key_to_index
    D = np.stack([U[i[b]] - U[i[a]] for a, b in pairs])
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    S = D @ D.T
    iu = np.triu_indices(len(D), 1)
    return float(S[iu].mean()), len(pairs)
