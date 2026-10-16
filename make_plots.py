"""Create the result plots from the JSON logs written by train.py / generate.py."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common as c

COLORS = {"SimpleRNN": "#0072B2", "BiRNN": "#D55E00", "LSTM": "#009E73", "GRU": "#CC79A7"}
PLOTS = os.path.join(c.RESULTS_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)


def load(name):
    with open(os.path.join(c.RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def training_curves():
    h = load("histories.json")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for kind, hist in h.items():
        ep = range(1, len(hist["loss"]) + 1)
        axes[0].plot(ep, hist["loss"], color=COLORS[kind], label=kind)
        axes[1].plot(ep, hist["val_loss"], color=COLORS[kind], label=kind)
        axes[2].plot(ep, hist["val_accuracy"], color=COLORS[kind], label=kind)
    style(axes[0], "Training loss", "Epoch", "Cross-entropy")
    style(axes[1], "Validation loss", "Epoch", "Cross-entropy")
    style(axes[2], "Validation next-word accuracy", "Epoch", "Accuracy")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "training_curves.png"), dpi=150)
    plt.close(fig)


def comparison_bars():
    m = load("metrics.json")
    kinds = list(m.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    spec = [("val_perplexity", "Validation perplexity (lower is better)", "Perplexity"),
            ("val_accuracy", "Validation accuracy (higher is better)", "Accuracy"),
            ("train_time_sec", "Total training time", "Seconds"),
            ("parameters", "Trainable parameters", "Count")]
    for ax, (key, title, yl) in zip(axes, spec):
        vals = [m[k][key] for k in kinds]
        bars = ax.bar(kinds, vals, color=[COLORS[k] for k in kinds])
        style(ax, title, "", yl)
        ax.tick_params(axis="x", labelsize=9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}" if v < 1000 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "model_comparison.png"), dpi=150)
    plt.close(fig)


def temperature_effects():
    g = load("generation_metrics.json")
    temps = c.TEMPERATURES
    spec = [("distinct2", "Diversity: distinct-2 (higher = more varied)"),
            ("repeated_trigram_ratio", "Repetition: repeated-trigram ratio (lower = better)"),
            ("bigram_validity", "Coherence proxy: bigrams seen in corpus")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axes, spec):
        for kind in c.MODEL_NAMES:
            ax.plot(temps, [g[kind][str(t)][key] for t in temps], marker="o",
                    color=COLORS[kind], label=kind)
        style(ax, title, "Sampling temperature", key)
        ax.set_xticks(temps)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "temperature_effects.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    training_curves()
    comparison_bars()
    temperature_effects()
    print("plots written to", PLOTS)
