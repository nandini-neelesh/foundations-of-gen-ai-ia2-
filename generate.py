"""Generate text from the four trained models with identical seeds and
several sampling temperatures; compute diversity / repetition / validity
metrics; save samples, metrics and plots."""
import os
import sys
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import common as c

N_SAMPLES = 8   # independent samples per (model, seed, temperature)


def sample_batch(model, seed_ids, temperature, n_samples, n_words, rng, unk_id):
    """Autoregressive generation: n_samples sequences in parallel."""
    ctx = np.tile(np.array(seed_ids[-c.SEQ_LEN:], dtype=np.int32), (n_samples, 1))
    out = np.zeros((n_samples, n_words), dtype=np.int32)
    for step in range(n_words):
        probs = model(ctx, training=False).numpy().astype("float64")
        probs[:, 0] = 0.0          # never emit <pad>
        probs[:, unk_id] = 0.0     # never emit <unk>
        probs = np.clip(probs, 1e-12, None)
        logits = np.log(probs) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        nxt = np.array([rng.choice(len(row), p=row) for row in p], dtype=np.int32)
        out[:, step] = nxt
        ctx = np.concatenate([ctx[:, 1:], nxt[:, None]], axis=1)
    return out


def text_metrics(tokens, corpus_bigrams):
    words = [t for t in tokens if t != "\n"]
    n = len(words)
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    distinct1 = len(set(words)) / n
    distinct2 = len(set(bigrams)) / max(len(bigrams), 1)
    rep3 = 1 - len(set(trigrams)) / max(len(trigrams), 1)   # repeated-trigram ratio
    immediate = sum(1 for a, b in bigrams if a == b) / max(len(bigrams), 1)
    valid_bi = sum(1 for b in bigrams if b in corpus_bigrams) / max(len(bigrams), 1)
    return dict(distinct1=distinct1, distinct2=distinct2, repeated_trigram_ratio=rep3,
                immediate_repeat_ratio=immediate, bigram_validity=valid_bi)


def main():
    data = c.prepare_data()
    itos, stoi = data["itos"], data["stoi"]
    unk_id = stoi["<unk>"]
    train_split = int(len(data["tokens"]) * (1 - c.VAL_FRACTION))
    train_tokens = data["tokens"][:train_split]
    words_only = [t for t in train_tokens if t != "\n"]
    corpus_bigrams = set(zip(words_only, words_only[1:]))

    seeds_ids = [list(c.encode(c.tokenize(s), stoi)) for s in c.SEED_TEXTS]
    seed_tokens = [[itos[i] for i in ids[-c.SEQ_LEN:]] for ids in seeds_ids]

    samples = {}       # model -> temp -> seed_idx -> list of strings
    metrics_rows = []  # per model/temp/seed averaged over samples
    for kind in c.MODEL_NAMES:
        model = tf.keras.models.load_model(os.path.join(c.MODELS_DIR, f"{kind}.keras"))
        samples[kind] = {}
        for T in c.TEMPERATURES:
            samples[kind][str(T)] = {}
            for si, ids in enumerate(seeds_ids):
                rng = np.random.default_rng(1000 + si)   # same RNG stream for every model
                gen = sample_batch(model, ids, T, N_SAMPLES, c.GEN_WORDS, rng, unk_id)
                texts, ms = [], []
                for row in gen:
                    toks = [itos[i] for i in row]
                    texts.append(c.detokenize(toks))
                    ms.append(text_metrics(toks, corpus_bigrams))
                samples[kind][str(T)][str(si)] = texts
                avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                metrics_rows.append(dict(model=kind, temperature=T, seed=si, **avg))
        print("generated:", kind, flush=True)

    c.save_json(samples, "generated_samples.json")
    c.save_json(metrics_rows, "generation_metrics_per_seed.json")

    # aggregate over seeds
    agg = {}
    for kind in c.MODEL_NAMES:
        agg[kind] = {}
        for T in c.TEMPERATURES:
            rows = [r for r in metrics_rows if r["model"] == kind and r["temperature"] == T]
            agg[kind][str(T)] = {k: float(np.mean([r[k] for r in rows]))
                                 for k in rows[0] if k not in ("model", "temperature", "seed")}
    c.save_json(agg, "generation_metrics.json")

    # human-readable sample file (first sample of each cell)
    lines = ["GENERATED TEXT SAMPLES (first of %d samples per cell, %d words)\n" % (N_SAMPLES, c.GEN_WORDS)]
    for si, st in enumerate(c.SEED_TEXTS):
        lines.append("=" * 90)
        lines.append("SEED %d: %s" % (si + 1, " ".join(seed_tokens[si]).replace("\n", " / ")))
        lines.append("=" * 90)
        for T in c.TEMPERATURES:
            for kind in c.MODEL_NAMES:
                lines.append("\n[%s | T=%s]" % (kind, T))
                lines.append(samples[kind][str(T)][str(si)][0])
        lines.append("")
    with open(os.path.join(c.RESULTS_DIR, "generated_samples.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("GENERATION DONE", flush=True)


if __name__ == "__main__":
    main()
