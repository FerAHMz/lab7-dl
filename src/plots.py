"""Graficas del laboratorio con un estilo comun (marcas delgadas, ejes y grid recesivos)."""
import matplotlib.pyplot as plt
import numpy as np

from .config import AG_CLASSES, FIG_DIR

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Paleta categorica validada para daltonismo
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#a15ccf", "#d6a228", "#d63a6b", "#4bb5c4", "#7a7a73"]
MODEL_COLORS = {"sgns": "#2a78d6", "gensim": "#eb6834", "glove": "#1baf7a"}
TSNE_COLORS = PALETTE + ["#5b3a29", "#9bbb3b", "#e377c2"]
INIT_COLORS = {"tfidf": "#7a7a73", "random": "#a15ccf", "sgns": "#2a78d6", "glove": "#1baf7a"}


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "legend.frameon": False,
    })


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")


def zipf(counts, slope, intercept, name="zipf"):
    r = np.arange(1, len(counts) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.loglog(r, counts, color=PALETTE[0], lw=1.4, label="corpus")
    ax.loglog(r, np.exp(intercept) * r ** slope, color=PALETTE[1], ls="--", lw=1.2,
              label=f"ajuste: pendiente {slope:.2f}")
    ax.loglog(r, counts[0] / r, color=MUTED, ls=":", lw=1, label="Zipf ideal (pendiente -1)")
    ax.set_xlabel("rango")
    ax.set_ylabel("frecuencia")
    ax.set_title("Frecuencia contra rango (log-log)")
    ax.legend()
    save(fig, name)
    return fig


def training_curves(runs, name, title):
    """Perdida por epoch y accuracy de analogias por epoch de varias iteraciones."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.1))
    for k, run in enumerate(runs):
        col = PALETTE[k % len(PALETTE)]
        h = run["history"]
        ep = [r["epoch"] for r in h]
        lbl = run.get("label", run["name"])
        steps = run.get("steps", [])
        if steps:
            a1.plot([s["step"] for s in steps], [s["loss"] for s in steps], color=col, lw=1.1, label=lbl)
        a2.plot(ep, [r["train_loss"] for r in h], color=col, marker="o", ms=3, label=lbl)
        a3.plot(ep, [r["an_total"] for r in h], color=col, marker="o", ms=3, label=lbl)
    a1.set(xlabel="paso", ylabel="pérdida (promedio cada 500 pasos)", title="Pérdida por pasos")
    a2.set(xlabel="epoch", ylabel="pérdida promedio", title="Pérdida por epoch")
    a3.set(xlabel="epoch", ylabel="accuracy total", title="Analogías (3CosAdd, top-30k)")
    a3.legend(fontsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, name)
    return fig


def analogy_vs_tokens(points, glove_acc, name="analogias_vs_tokens"):
    """points: lista de (tokens, sem, syn, total, etiqueta)."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    pts = sorted(points)
    x = [p[0] / 1e6 for p in pts]
    for j, (lbl, col) in enumerate([("semánticas", PALETTE[0]), ("sintácticas", PALETTE[1]),
                                    ("total", INK)]):
        ax.plot(x, [p[1 + j] for p in pts], marker="o", ms=4, color=col, label=f"SGNS {lbl}")
    for p in pts:
        ax.annotate(p[4], (p[0] / 1e6, p[3]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axhline(glove_acc, color=PALETTE[2], ls="--", lw=1.2, label=f"GloVe 6B (total {glove_acc:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{v:.1f}M" for v in x])
    ax.minorticks_off()
    ax.set_xlabel("tokens del corpus de entrenamiento (escala log)")
    ax.set_ylabel("accuracy de analogías")
    ax.set_title("Analogías contra tamaño del corpus")
    ax.legend(fontsize=7)
    save(fig, name)
    return fig


def classifier_curves(results, name="clasificacion_curvas"):
    ncol = 4
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.5, 2.5 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(results):]:
        ax.set_visible(False)
    for ax, (lbl, r) in zip(axes, results.items()):
        ep = [h["epoch"] for h in r["history"]]
        ax.plot(ep, [h["train_loss"] for h in r["history"]], color=PALETTE[0], label="train")
        ax.plot(ep, [h["val_loss"] for h in r["history"]], color=PALETTE[0], ls="--", label="val")
        ax.axvline(r["best"]["epoch"], color=MUTED, lw=0.8, ls=":")
        ax.set_title(lbl, fontsize=8.5)
        ax.set_xlabel("epoch")
    for ax in axes[::ncol]:
        ax.set_ylabel("cross-entropy")
    axes[0].legend()
    fig.tight_layout()
    save(fig, name)
    return fig


def confusion_matrices(cms, titles, name="matrices_confusion"):
    fig, axes = plt.subplots(1, len(cms), figsize=(3.0 * len(cms), 3.0))
    for ax, cm, t in zip(np.atleast_1d(axes), cms, titles):
        cm = np.asarray(cm)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), AG_CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(4), AG_CLASSES)
        ax.set_xlabel("predicción")
        ax.set_title(t, fontsize=8.5)
    np.atleast_1d(axes)[0].set_ylabel("real")
    fig.tight_layout()
    save(fig, name)
    return fig


def f1_vs_fraction(table, name="f1_vs_fraccion"):
    """table: DataFrame con columnas modelo, frac, f1_test."""
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    styles = {"congelado": "--", "fine-tuning": "-"}
    for k, (model, g) in enumerate(table.groupby("modelo", sort=False)):
        g = g.sort_values("frac")
        init = model.split()[0].lower().replace("-", "").replace("aleatorio", "random")
        col = INIT_COLORS.get(init, PALETTE[k])
        ls = "--" if "congel" in model else "-"
        ax.plot(100 * g["frac"], g["f1_test"], marker="o", ms=4, color=col, ls=ls, label=model)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 50, 100], ["1 %", "10 %", "50 %", "100 %"])
    ax.minorticks_off()
    ax.set_xlabel("datos de entrenamiento de AG News (escala log)")
    ax.set_ylabel("F1 macro en test")
    ax.set_title("F1 de test contra cantidad de datos")
    ax.legend(fontsize=7)
    save(fig, name)
    return fig


def tsne(xy, groups, labels, name="tsne"):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    names = list(dict.fromkeys(groups))
    for k, g in enumerate(names):
        sel = np.array([x == g for x in groups])
        ax.scatter(xy[sel, 0], xy[sel, 1], s=10, color=TSNE_COLORS[k % len(TSNE_COLORS)], label=g, alpha=0.85)
    for i in range(0, len(labels), 4):
        ax.annotate(labels[i], xy[i], fontsize=5.5, color=MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    ax.set_title("t-SNE de ~500 palabras por grupo temático (SGNS final)")
    save(fig, name)
    return fig
