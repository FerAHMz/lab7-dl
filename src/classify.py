"""Pipeline end-to-end en AG News: texto -> tokens -> IDs -> EmbeddingBag + MLP -> clase.

El preprocesamiento y el tokenizador son los mismos de la seccion 2 (src.text.tokenize).
"""
import copy
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .config import SEED, VAL_FRACTION
from .text import tokenize

PAD, UNK = "<pad>", "<unk>"


# ----------------------------------------------------------------------------- datos
def load_ag_news():
    """Tokeniza AG News y separa 10 % estratificado del train oficial como validacion."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    tr_text, tr_y = ds["train"]["text"], np.array(ds["train"]["label"])
    te_text, te_y = ds["test"]["text"], np.array(ds["test"]["label"])
    idx_tr, idx_va = train_test_split(np.arange(len(tr_y)), test_size=VAL_FRACTION,
                                      stratify=tr_y, random_state=SEED)
    toks_tr_all = [tokenize(t) for t in tr_text]
    toks_te = [tokenize(t) for t in te_text]
    split = {
        "train": ([toks_tr_all[i] for i in idx_tr], tr_y[idx_tr], [tr_text[i] for i in idx_tr]),
        "val": ([toks_tr_all[i] for i in idx_va], tr_y[idx_va], [tr_text[i] for i in idx_va]),
        "test": (toks_te, te_y, list(te_text)),
    }
    return split


def stratified_fraction(y, frac, seed=SEED):
    """Indices de una submuestra estratificada del set de entrenamiento."""
    if frac >= 1.0:
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=frac, stratify=y, random_state=seed)
    return np.sort(idx)


def oov_rate(token_lists, vocab):
    """% de tokens (ocurrencias) y de tipos fuera del vocabulario de un conjunto de embeddings."""
    c = Counter(t for toks in token_lists for t in toks)
    n_tok = sum(c.values())
    oov_tok = sum(v for w, v in c.items() if w not in vocab)
    oov_types = sum(1 for w in c if w not in vocab)
    return 100 * oov_tok / n_tok, 100 * oov_types / len(c)


# ----------------------------------------------------------------------------- vocabulario
def vocab_from_corpus(token_lists, min_count=2):
    """Vocabulario aprendido en la tarea (embeddings aleatorios): <pad>, <unk> y palabras >= min_count."""
    c = Counter(t for toks in token_lists for t in toks)
    words = [PAD, UNK] + [w for w, n in c.most_common() if n >= min_count]
    return {w: i for i, w in enumerate(words)}, True


def vocab_from_embeddings(kv, token_lists):
    """Vocabulario para embeddings preentrenados: las palabras de AG News que el modelo conoce.

    Los tokens fuera del vocabulario del embedding se mapean a <pad>, que EmbeddingBag ignora
    al promediar (padding_idx), asi que el documento se representa solo con palabras conocidas.
    """
    seen = {t for toks in token_lists for t in toks}
    words = [PAD] + [w for w in kv.index_to_key if w in seen]
    return {w: i for i, w in enumerate(words)}, False


def encode(token_lists, w2i, has_unk):
    """Lista de documentos -> (IDs concatenados, offsets) para nn.EmbeddingBag."""
    unk = w2i.get(UNK, 0) if has_unk else 0
    ids, offsets = [], [0]
    for toks in token_lists:
        doc = [w2i.get(t, unk) for t in toks] or [0]
        ids.extend(doc)
        offsets.append(offsets[-1] + len(doc))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def embedding_matrix(kv, w2i, dim):
    W = torch.zeros(len(w2i), dim)
    for w, i in w2i.items():
        if i > 0:
            W[i] = torch.from_numpy(kv[w].copy())
    return W


# ----------------------------------------------------------------------------- modelo
class BagClassifier(nn.Module):
    """nn.EmbeddingBag (promedio) + MLP de una capa oculta."""

    def __init__(self, vocab_size, dim, n_classes=4, hidden=256, dropout=0.3,
                 pretrained=None, freeze=False):
        super().__init__()
        if pretrained is not None:
            self.emb = nn.EmbeddingBag.from_pretrained(pretrained, freeze=freeze, mode="mean",
                                                       padding_idx=0)
        else:
            self.emb = nn.EmbeddingBag(vocab_size, dim, mode="mean", padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, ids, offsets):
        return self.mlp(self.emb(ids, offsets))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"acc": accuracy_score(y, p), "precision": pr, "recall": rc, "f1": f1}


class Batches:
    """Minibatches de documentos codificados (ids + offsets) en el dispositivo."""

    def __init__(self, token_lists, y, w2i, has_unk, batch_size, shuffle, device):
        self.docs = [torch.tensor([w2i.get(t, w2i.get(UNK, 0) if has_unk else 0) for t in toks] or [0])
                     for toks in token_lists]
        self.y = torch.tensor(y, dtype=torch.long)
        self.bs, self.shuffle, self.device = batch_size, shuffle, device

    def __len__(self):
        return (len(self.docs) + self.bs - 1) // self.bs

    def __iter__(self):
        order = torch.randperm(len(self.docs)) if self.shuffle else torch.arange(len(self.docs))
        for s in range(0, len(order), self.bs):
            idx = order[s:s + self.bs]
            docs = [self.docs[i] for i in idx]
            lengths = torch.tensor([len(d) for d in docs])
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)[:-1]])
            yield (torch.cat(docs).to(self.device), offsets.to(self.device), self.y[idx].to(self.device))


@torch.no_grad()
def predict(model, batches):
    model.eval()
    loss, preds, ys = 0.0, [], []
    for ids, off, yb in batches:
        out = model(ids, off)
        loss += F.cross_entropy(out, yb, reduction="sum").item()
        preds.append(out.argmax(1).cpu())
        ys.append(yb.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(preds).numpy()
    return loss / len(y), y, p


def train_classifier(data, init, kv=None, freeze=False, frac=1.0, dim=100, epochs=None,
                     lr=1e-3, batch_size=256, patience=3, device="cpu", seed=SEED, verbose=False):
    """Entrena el clasificador con early stopping sobre el F1 macro de validacion.

    init: "random" | "pretrained" (con kv). Devuelve historial, mejor epoch, metricas de val,
    parametros, tiempo y el modelo con los mejores pesos.
    """
    torch.manual_seed(seed)
    tr_toks, tr_y, _ = data["train"]
    sel = stratified_fraction(tr_y, frac)
    tr_toks, tr_y = [tr_toks[i] for i in sel], tr_y[sel]
    if init == "random":
        w2i, has_unk = vocab_from_corpus(tr_toks)
        model = BagClassifier(len(w2i), dim)
    else:
        all_toks = data["train"][0] + data["val"][0] + data["test"][0]
        w2i, has_unk = vocab_from_embeddings(kv, all_toks)
        dim = kv.vector_size
        model = BagClassifier(len(w2i), dim, pretrained=embedding_matrix(kv, w2i, dim), freeze=freeze)
    model.to(device)
    tr = Batches(tr_toks, tr_y, w2i, has_unk, batch_size, True, device)
    va = Batches(*data["val"][:2], w2i, has_unk, 1024, False, device)
    # con pocos datos se necesitan mas epochs para dar un numero razonable de pasos
    if epochs is None:
        epochs = int(min(100, max(20, np.ceil(3000 / len(tr)))))
        patience = max(patience, epochs // 10)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    total, trainable = count_params(model)

    history, best, best_state, bad = [], None, None, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for ids, off, yb in tr:
            loss = F.cross_entropy(model(ids, off), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            n += len(yb)
        val_loss, y, p = predict(model, va)
        m = metrics(y, p)
        history.append({"epoch": ep, "train_loss": run_loss / n, "val_loss": val_loss, **m})
        if verbose:
            print(f"  ep {ep:3d} train {run_loss / n:.4f} val {val_loss:.4f} f1 {m['f1']:.4f}")
        if best is None or m["f1"] > best["f1"]:
            best, best_state, bad = {"epoch": ep, **m}, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return {"init": init, "freeze": freeze, "frac": frac, "n_train": len(tr_y), "history": history,
            "best": best, "params_total": total, "params_trainable": trainable,
            "train_time_s": train_time, "vocab": len(w2i), "w2i": w2i, "has_unk": has_unk,
            "model": model}


def evaluate_test(res, data, device="cpu"):
    te = Batches(*data["test"][:2], res["w2i"], res["has_unk"], 1024, False, device)
    _, y, p = predict(res["model"], te)
    return {**metrics(y, p), "cm": confusion_matrix(y, p).tolist()}


# ----------------------------------------------------------------------------- baseline
def tfidf_baseline(data, frac=1.0, Cs=(1.0, 4.0, 16.0)):
    """TF-IDF (unigramas + bigramas, mismo tokenizador) + regresion logistica; C elegido en val."""
    tr_toks, tr_y, _ = data["train"]
    sel = stratified_fraction(tr_y, frac)
    tr_toks, tr_y = [tr_toks[i] for i in sel], tr_y[sel]
    vec = TfidfVectorizer(analyzer=lambda toks: toks + [a + " " + b for a, b in zip(toks, toks[1:])],
                          min_df=2 if frac >= 0.1 else 1, sublinear_tf=True)
    t0 = time.perf_counter()
    Xtr = vec.fit_transform(tr_toks)
    Xva = vec.transform(data["val"][0])
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Xtr, tr_y)
        m = metrics(data["val"][1], clf.predict(Xva))
        if best is None or m["f1"] > best[1]["f1"]:
            best = (clf, m, C)
    train_time = time.perf_counter() - t0
    clf, m, C = best
    n_params = clf.coef_.size + clf.intercept_.size
    return {"init": "tfidf", "freeze": False, "frac": frac, "n_train": len(tr_y), "best": m, "C": C,
            "params_total": n_params, "params_trainable": n_params, "train_time_s": train_time,
            "vocab": len(vec.vocabulary_), "vec": vec, "clf": clf}


def tfidf_test(res, data):
    p = res["clf"].predict(res["vec"].transform(data["test"][0]))
    return {**metrics(data["test"][1], p), "cm": confusion_matrix(data["test"][1], p).tolist()}
