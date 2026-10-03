"""Skip-gram con negative sampling (SGNS) en PyTorch, con dos tablas nn.Embedding.

Por epoch: submuestreo de palabras frecuentes -> ventana dinamica por palabra central ->
pares (centro, contexto) generados en la GPU -> barajado -> minibatches con k negativos
muestreados de la distribucion unigrama^0.75.
"""
import gc
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RUNS_DIR, VECTORS_DIR
from .evaluation import quick_eval, to_keyed_vectors
from .text import keep_probability


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class PeakMemory:
    """Memoria pico del acelerador.

    CUDA: torch.cuda.max_memory_allocated(). MPS no tiene contador de pico, asi que se muestrea
    torch.mps.current_allocated_memory() en los puntos de mayor uso (pares del epoch generados y
    despues de cada backward).
    """

    def __init__(self, device):
        self.device, self.peak = device, 0
        free_memory(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        if self.device.type == "mps":
            self.peak = max(self.peak, torch.mps.current_allocated_memory())

    def value_mb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2**20
        return self.peak / 2**20


class SGNS(nn.Module):
    """W (entrada, palabra central) y W' (salida, contexto), inicializadas como word2vec."""

    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim)
        self.out = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                                   # B x D
        u = self.out(context)                                  # B x D
        un = self.out(negatives)                               # B x K x D
        pos = F.logsigmoid((v * u).sum(-1))                    # log s(u_o . v_c)
        neg = F.logsigmoid(-torch.bmm(un, v.unsqueeze(2)).squeeze(2)).sum(-1)  # sum log s(-u_k . v_c)
        return -(pos + neg).mean()


def make_pairs(ids, para, keep_prob, window, device):
    """Pares (centro, contexto) de un epoch, todo en el dispositivo.

    1) Submuestreo: cada token se conserva con probabilidad keep_prob[id] (antes de formar ventanas,
       como word2vec, asi que la ventana efectiva se agranda).
    2) Ventana dinamica: para cada centro se muestrea b ~ U{1..window} y se usan los contextos a
       distancia d <= b (las palabras cercanas pesan mas).
    3) Las ventanas no cruzan limites de parrafo.
    """
    keep = torch.rand(len(ids), device=device) < keep_prob[ids]
    ids_s, para_s = ids[keep], para[keep]
    n = len(ids_s)
    b = torch.randint(1, window + 1, (n,), device=device)
    centers, contexts = [], []
    for d in range(1, window + 1):
        same = para_s[:-d] == para_s[d:]
        right = (same & (b[:-d] >= d)).nonzero().squeeze(1)    # centro i, contexto i+d
        left = (same & (b[d:] >= d)).nonzero().squeeze(1)      # centro i+d, contexto i
        centers += [ids_s[right], ids_s[left + d]]
        contexts += [ids_s[right + d], ids_s[left]]
    return torch.cat(centers), torch.cat(contexts), n


def unigram_table(counts, power=0.75, size=20_000_000, device="cpu"):
    """Tabla de muestreo de negativos: cada id aparece proporcional a count^0.75."""
    p = counts.astype(np.float64) ** power
    p /= p.sum()
    reps = np.round(p * size).astype(np.int64)
    reps[reps == 0] = 1
    return torch.tensor(np.repeat(np.arange(len(counts), dtype=np.int32), reps), device=device)


def train_sgns(corpus, cfg, device=None, log_every=500, verbose=True):
    """Entrena SGNS con la configuracion `cfg` y registra todo lo que pide la seccion 4.

    cfg: name, dim, window, negatives, sample (umbral t), min_count, lr, epochs, batch_size,
         fraction (del subconjunto).
    """
    device = device or get_device()
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))
    corpus = corpus.fraction(cfg["fraction"]) if cfg.get("fraction", 1.0) < 1.0 else corpus
    corpus = corpus.restrict(cfg["min_count"])
    V, D, K, B = len(corpus.words), cfg["dim"], cfg["negatives"], cfg["batch_size"]

    ids = torch.tensor(corpus.ids, device=device)
    para = torch.tensor(corpus.para_ids, device=device)
    keep_prob = torch.tensor(keep_probability(corpus.counts, cfg["sample"]), dtype=torch.float32, device=device)
    table = unigram_table(corpus.counts, device=device)

    model = SGNS(V, D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    mem = PeakMemory(device)

    history, steps_log, t_total = [], [], 0.0
    # pares esperados por epoch (para el decaimiento lineal del learning rate, como word2vec)
    est_pairs = None
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        sync(device)
        t0 = time.perf_counter()
        centers, contexts, n_sub = make_pairs(ids, para, keep_prob, cfg["window"], device)
        P = len(centers)
        perm = torch.randperm(P, device=device)
        centers, contexts = centers[perm], contexts[perm]
        del perm
        mem.sample()
        if est_pairs is None:
            est_pairs = P
            total_steps = cfg["epochs"] * ((P + B - 1) // B)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: max(1e-4, 1 - s / total_steps))
        model.train()
        loss_sum, loss_n = torch.zeros((), device=device), 0
        window_sum, window_n = torch.zeros((), device=device), 0
        for s in range(0, P, B):
            c, o = centers[s:s + B], contexts[s:s + B]
            neg = table[torch.randint(len(table), (len(c), K), device=device)]
            loss = model(c, o, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            loss_sum += loss.detach()
            loss_n += 1
            window_sum += loss.detach()
            window_n += 1
            if step % log_every == 0:
                mem.sample()
                steps_log.append({"step": step, "epoch": epoch, "loss": (window_sum / window_n).item()})
                window_sum, window_n = torch.zeros((), device=device), 0
        sync(device)
        t_epoch = time.perf_counter() - t0
        t_total += t_epoch
        del centers, contexts
        free_memory(device)

        kv = to_keyed_vectors(corpus.words, model.inp.weight.detach().cpu().numpy())
        ev = quick_eval(kv, device=device)
        row = {"epoch": epoch, "train_loss": (loss_sum / loss_n).item(), "pairs": P,
               "tokens_after_subsampling": n_sub, "time_s": t_epoch, **ev}
        history.append(row)
        if verbose:
            print(f"[{cfg['name']}] epoch {epoch}/{cfg['epochs']}  loss {row['train_loss']:.4f}  "
                  f"analogias sem {ev['an_sem']:.3f} sin {ev['an_syn']:.3f} total {ev['an_total']:.3f}  "
                  f"wordsim {ev['wordsim']:.3f}  ({t_epoch:.0f}s, {P / 1e6:.1f}M pares)")

    result = {
        "name": cfg["name"], "cfg": cfg, "impl": "sgns-pytorch",
        "corpus_tokens": int(corpus.n_tokens), "vocab": V,
        "history": history, "steps": steps_log,
        "time_total_s": t_total, "peak_mem_mb": mem.value_mb(), "device": str(device),
        "params": sum(p.numel() for p in model.parameters()),
    }
    kv = to_keyed_vectors(corpus.words, model.inp.weight.detach().cpu().numpy())
    del model, opt, ids, para, table
    free_memory(device)
    return result, kv


def save_run(result, kv):
    with open(RUNS_DIR / f"{result['name']}.json", "w") as f:
        json.dump(result, f, indent=1, default=float)
    kv.save(str(VECTORS_DIR / f"{result['name']}.kv"))


def load_run(name):
    from gensim.models import KeyedVectors
    path = RUNS_DIR / f"{name}.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        result = json.load(f)
    kv_path = VECTORS_DIR / f"{name}.kv"
    kv = KeyedVectors.load(str(kv_path)) if kv_path.exists() else None
    return result, kv


def run_or_load(corpus, cfg, device=None, **kw):
    """Entrena la iteracion o la carga de results/ si ya existe (para re-ejecutar el notebook)."""
    result, kv = load_run(cfg["name"])
    if result is not None and kv is not None:
        print(f"[{cfg['name']}] cargada de results/runs (entrenada en {result['time_total_s'] / 60:.1f} min)")
        return result, kv
    result, kv = train_sgns(corpus, cfg, device, **kw)
    save_run(result, kv)
    return result, kv


def train_gensim(corpus, cfg, workers=12):
    """Word2Vec de gensim (skip-gram, negative sampling) sobre el mismo corpus y vocabulario."""
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec

    corpus = corpus.fraction(cfg["fraction"]) if cfg.get("fraction", 1.0) < 1.0 else corpus
    corpus = corpus.restrict(cfg["min_count"])
    sentences = corpus.sentences()

    class EpochLogger(CallbackAny2Vec):
        def __init__(self):
            self.history, self.prev_loss, self.t0, self.epoch = [], 0.0, None, 0

        def on_epoch_begin(self, model):
            self.t0 = time.perf_counter()

        def on_epoch_end(self, model):
            self.epoch += 1
            dt = time.perf_counter() - self.t0
            cum = model.get_latest_training_loss()
            model.wv.fill_norms(force=True)
            ev = quick_eval(model.wv)
            self.history.append({"epoch": self.epoch, "train_loss_sum": cum - self.prev_loss,
                                 "time_s": dt, **ev})
            self.prev_loss = cum
            print(f"[{cfg['name']}] epoch {self.epoch}  analogias total {ev['an_total']:.3f}  "
                  f"wordsim {ev['wordsim']:.3f}  ({dt:.0f}s)")

    logger = EpochLogger()
    model = Word2Vec(vector_size=cfg["dim"], window=cfg["window"], negative=cfg["negatives"],
                     sample=cfg["sample"], min_count=cfg["min_count"], alpha=cfg.get("alpha", 0.025),
                     sg=1, hs=0, workers=workers, seed=cfg.get("seed", 42), compute_loss=True)
    model.build_vocab(sentences)
    t0 = time.perf_counter()
    model.train(sentences, total_examples=len(sentences), epochs=cfg["epochs"],
                compute_loss=True, callbacks=[logger])
    t_total = time.perf_counter() - t0
    # KeyedVectors en el mismo orden (frecuencia descendente) que el SGNS propio
    kv = to_keyed_vectors(model.wv.index_to_key, model.wv.vectors)
    result = {"name": cfg["name"], "cfg": cfg, "impl": "gensim-word2vec",
              "corpus_tokens": int(corpus.n_tokens), "vocab": len(kv), "history": logger.history,
              "time_total_s": t_total, "workers": workers,
              "vocab_identical": model.wv.index_to_key == corpus.words or
              set(model.wv.index_to_key) == set(corpus.words)}
    return result, kv
