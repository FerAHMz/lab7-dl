"""Primeras etapas del pipeline: texto crudo -> texto normalizado -> tokens -> IDs.

La misma normalizacion y el mismo tokenizador se usan para WikiText-103 (entrenamiento de
embeddings) y para AG News (clasificacion), y estan alineados con la tokenizacion de GloVe 6B
(minusculas, clitics separados al estilo Penn Treebank, palabras con guion y acronimos con punto).
"""
import html
import re
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from .config import PROC_DIR, SUBSET_TOKENS

# ----------------------------------------------------------------------------- normalizacion
# WikiText escribe los separadores dentro de palabras y numeros como " @-@ ", " @,@ ", " @.@ "
_WIKI_JOINERS = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")]
_CLITIC_NT = re.compile(r"(?<=\w)n't\b")                      # don't -> do n't (como GloVe / PTB)
_WIKI_NT = re.compile(r"(?<=\w)n 't\b")                      # WikiText escribe "wasn 't" -> "was n't"
_CLITIC = re.compile(r"(?<=\w)'(s|re|ve|ll|d|m)\b")           # game's -> game 's
_AG_ARTIFACTS = re.compile(r"\\+|#\d+;|&\w+;|\b(?:quot|amp|lt|gt);")  # "\" (salto de linea), "#39;", "quot;" en AG News
_HTML_TAG = re.compile(r"<[^>]{0,300}>")
_SPACES = re.compile(r"\s+")

TOKEN_RE = re.compile(r"""
      (?:[^\W\d_]\.){2,}             # acronimos con punto: u.s.  u.k.  e.g.
    | n't | '(?:s|re|ve|ll|d|m)\b    # clitics separados
    | \d+(?:[.,]\d+)+                # numeros con formato: 3.5  1,000
    | [^\W_]+(?:[-'][^\W_]+)*        # palabras (letras/digitos), con guion o apostrofo interno
""", re.VERBOSE)


def normalize(text):
    """Minusculas, reconstruccion de separadores de WikiText y limpieza de artefactos HTML."""
    text = html.unescape(text.replace("#39;", "'").replace("#36;", "$"))
    text = _HTML_TAG.sub(" ", text)
    text = _AG_ARTIFACTS.sub(" ", text)
    for a, b in _WIKI_JOINERS:
        text = text.replace(a, b)
    text = text.lower().replace("\u2019", "'")
    text = _WIKI_NT.sub(" n't", text)
    text = _CLITIC_NT.sub(" n't", text)
    text = _CLITIC.sub(r" '\1", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text):
    """Tokenizador propio por palabra: normaliza y extrae tokens; descarta puntuacion."""
    return TOKEN_RE.findall(normalize(text))


# ----------------------------------------------------------------------------- WikiText-103
_ARTICLE = re.compile(r"^ = [^=].* = $")
_HEADING = re.compile(r"^ (= ){2,}.*( =){2,} $")


def line_kind(line):
    s = line.rstrip("\n")
    if not s.strip():
        return "empty"
    if _ARTICLE.match(s):
        return "article"
    if _HEADING.match(s):
        return "heading"
    return "text"


def classify_lines(lines):
    """Tipo de cada linea. Un titulo de articulo (` = Titulo = `) va entre lineas vacias; las
    lineas con ese formato dentro de tablas (` = Goals ; A = `) se tratan como texto."""
    kinds = [line_kind(l) for l in lines]
    for i, k in enumerate(kinds):
        if k == "article":
            prev_empty = i == 0 or kinds[i - 1] == "empty"
            next_empty = i + 1 == len(kinds) or kinds[i + 1] == "empty"
            if not (prev_empty and next_empty):
                kinds[i] = "text"
    return kinds


def _scan_chunk(block):
    """Worker: cuenta tokens por linea y frecuencias de palabras de un bloque de lineas."""
    counter, per_line = Counter(), []
    for line, kind in zip(*block):
        if kind == "text":
            toks = tokenize(line)
            counter.update(toks)
            per_line.append(len(toks))
        else:
            per_line.append(0)
    return counter, per_line


def load_wikitext(split="train"):
    from datasets import load_dataset
    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")[split]["text"]


def scan_corpus(lines, workers=12, chunk=50_000):
    """Estadisticas del corpus completo: articulos, lineas, tokens y frecuencias (en paralelo)."""
    kinds = classify_lines(lines)
    blocks = [(lines[i:i + chunk], kinds[i:i + chunk]) for i in range(0, len(lines), chunk)]
    with Pool(workers) as pool:
        parts = pool.map(_scan_chunk, blocks)
    counter, per_line = Counter(), []
    for c, p in parts:
        counter.update(c)
        per_line.extend(p)
    return {"kinds": Counter(kinds), "counter": counter, "tokens_per_line": np.array(per_line, dtype=np.int64)}


# ----------------------------------------------------------------------------- corpus en IDs
@dataclass
class Corpus:
    """Corpus como arreglo plano de IDs mas el inicio de cada parrafo.

    Los IDs estan ordenados por frecuencia descendente, asi que aplicar un min_count equivale a
    quedarse con los IDs < V; las ventanas de contexto nunca cruzan el limite de un parrafo.
    """
    ids: np.ndarray        # int32, tokens del corpus
    starts: np.ndarray     # int64, posicion inicial de cada parrafo (len = n_parrafos + 1)
    words: list            # id -> palabra
    counts: np.ndarray     # frecuencia de cada id

    @property
    def n_tokens(self):
        return len(self.ids)

    @property
    def para_ids(self):
        lengths = np.diff(self.starts)
        return np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)

    def restrict(self, min_count):
        """Elimina del flujo los tokens con frecuencia < min_count (como word2vec/gensim)."""
        V = int((self.counts >= min_count).sum())
        keep = self.ids < V
        ck = np.concatenate([[0], np.cumsum(keep)])
        starts = ck[self.starts]
        return Corpus(self.ids[keep], starts, self.words[:V], self.counts[:V])

    def fraction(self, frac):
        """Prefijo del corpus (parrafos completos) con ~frac de los tokens; recalcula frecuencias."""
        cut = int(np.searchsorted(self.starts, frac * self.n_tokens))
        ids = self.ids[: self.starts[cut]]
        counts = np.bincount(ids, minlength=len(self.words))
        order = np.argsort(-counts, kind="stable")
        order = order[counts[order] > 0]
        remap = np.full(len(self.words), -1, dtype=np.int32)
        remap[order] = np.arange(len(order), dtype=np.int32)
        return Corpus(remap[ids], self.starts[: cut + 1].copy(),
                      [self.words[i] for i in order], counts[order])

    def sentences(self, words=True):
        """Parrafos como listas de palabras (entrada para gensim)."""
        out = []
        for s, e in zip(self.starts[:-1], self.starts[1:]):
            if e > s:
                seg = self.ids[s:e]
                out.append([self.words[i] for i in seg] if words else seg)
        return out


def _tokenize_lines(lines):
    return [tokenize(l) for l in lines]


def build_subset(lines, tokens_per_line, target=SUBSET_TOKENS, workers=12):
    """Prefijo de articulos completos de WikiText-103 con al menos `target` tokens."""
    kinds = classify_lines(lines)
    article_starts = [i for i, k in enumerate(kinds) if k == "article"] + [len(lines)]
    cum = np.cumsum(tokens_per_line)
    n_art = next(a for a, s in enumerate(article_starts[1:], 1) if cum[s - 1] >= target)
    end = article_starts[n_art]
    text_lines = [l for l, k in zip(lines[:end], kinds[:end]) if k == "text"]
    blocks = [text_lines[i:i + 20_000] for i in range(0, len(text_lines), 20_000)]
    with Pool(workers) as pool:
        paras = [p for part in pool.map(_tokenize_lines, blocks) for p in part if p]
    counter = Counter()
    for p in paras:
        counter.update(p)
    words = [w for w, _ in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
    w2i = {w: i for i, w in enumerate(words)}
    ids = np.fromiter((w2i[t] for p in paras for t in p), dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum([len(p) for p in paras])]).astype(np.int64)
    counts = np.array([counter[w] for w in words], dtype=np.int64)
    return Corpus(ids, starts, words, counts), n_art


def save_corpus(corpus, name="wikitext_subset"):
    np.savez_compressed(PROC_DIR / f"{name}.npz", ids=corpus.ids, starts=corpus.starts,
                        counts=corpus.counts, words=np.array(corpus.words, dtype=object))


def load_corpus(name="wikitext_subset"):
    path = PROC_DIR / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return Corpus(z["ids"], z["starts"], list(z["words"]), z["counts"])


# ----------------------------------------------------------------------------- estadisticas
def zipf_fit(counts, lo=10, hi=10_000):
    """Pendiente de log(frecuencia) vs log(rango) en el tramo central (ley de Zipf: ~ -1)."""
    r = np.arange(1, len(counts) + 1)
    sel = (r >= lo) & (r <= hi)
    slope, intercept = np.polyfit(np.log(r[sel]), np.log(counts[sel]), 1)
    return slope, intercept


def coverage(counts, ks=(10, 1_000, 30_000)):
    total = counts.sum()
    return {k: counts[:k].sum() / total for k in ks}


def vocab_table(counts, thresholds=(1, 2, 5, 10, 20, 50, 100)):
    total = counts.sum()
    rows = []
    for m in thresholds:
        keep = counts >= m
        rows.append({"min_count": m, "vocab": int(keep.sum()),
                     "tokens_unk_%": 100 * counts[~keep].sum() / total})
    return rows


def keep_probability(counts, t=1e-4):
    """Probabilidad de conservar cada palabra (formula del codigo C de word2vec y de gensim).

    p_keep(w) = (sqrt(f/t) + 1) * t / f,  con f = frecuencia relativa; se recorta a 1.
    El paper (Mikolov et al., 2013) usa p_discard = 1 - sqrt(t/f), algo mas agresivo.
    """
    f = counts / counts.sum()
    return np.minimum(1.0, (np.sqrt(f / t) + 1) * t / f)


def count_pairs(starts, window):
    """Pares (centro, contexto) con ventana fija `window` en parrafos de longitud L."""
    L = np.diff(starts)
    total = 0
    for d in range(1, window + 1):
        total += 2 * np.clip(L - d, 0, None).sum()
    return int(total)


def expected_pairs_dynamic(starts, window):
    """Pares esperados con ventana dinamica (b ~ U{1..window}), como word2vec/gensim."""
    L = np.diff(starts)
    total = 0.0
    for d in range(1, window + 1):
        p = (window - d + 1) / window            # P(b >= d)
        total += 2 * p * np.clip(L - d, 0, None).sum()
    return int(total)


def prepare_wikitext(workers=12, force=False):
    """Escanea WikiText-103 (train) y construye el subconjunto en IDs; ambos se cachean en disco."""
    import pickle
    stats_path = PROC_DIR / "wikitext_stats.pkl"
    corpus = load_corpus()
    if not force and stats_path.exists() and corpus is not None:
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        return stats, corpus
    lines = load_wikitext("train")
    stats = scan_corpus(lines, workers)
    corpus, n_art = build_subset(lines, stats["tokens_per_line"], workers=workers)
    stats["subset_articles"] = n_art
    stats["n_lines"] = len(lines)
    save_corpus(corpus)
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    return stats, corpus
