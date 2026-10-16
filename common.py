"""Shared preprocessing and model-building code for the autoregressive
word-level text-generation experiments (Simple RNN, BiRNN, LSTM, GRU)."""
import os
import re
import json
import random
from collections import Counter

import numpy as np

SEED = 42
SEQ_LEN = 20          # words of context used to predict the next word
MIN_FREQ = 5          # words rarer than this are mapped to <unk>
EMBED_DIM = 64
HIDDEN_UNITS = 128    # units per recurrent direction
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.10   # last 10% of the corpus is held out (no leakage)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "input.txt")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAMES = ["SimpleRNN", "BiRNN", "LSTM", "GRU"]

# Seeds shared by every model during generation
# (the model conditions on the last SEQ_LEN tokens of each seed)
SEED_TEXTS = [
    "first citizen :\nbefore we proceed any further , hear me speak .\n\nall :\nspeak , speak .",
    "hamlet :\nto be , or not to be , that is the question :\nwhether 'tis nobler in the mind to suffer",
    "romeo :\nwhat light through yonder window breaks ?\nit is the east , and juliet is the sun .\n\njuliet :",
]
TEMPERATURES = [0.5, 1.0, 1.5]
GEN_WORDS = 60


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def tokenize(text):
    """Lower-case; keep punctuation and line breaks as separate tokens."""
    text = text.lower()
    text = text.replace("\n", " \n ")
    tokens = re.findall(r"\n|[a-z]+(?:'[a-z]+)?|[.,;:!?]", text)
    return tokens


def load_corpus():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    return raw, tokenize(raw)


def build_vocab(train_tokens):
    """Vocabulary built from the TRAIN split only."""
    counts = Counter(train_tokens)
    words = sorted(w for w, c in counts.items() if c >= MIN_FREQ)
    itos = ["<pad>", "<unk>"] + words
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi, counts


def encode(tokens, stoi):
    unk = stoi["<unk>"]
    return np.array([stoi.get(t, unk) for t in tokens], dtype=np.int32)


def make_sequences(ids, seq_len=SEQ_LEN):
    """Sliding window (stride 1): X[i] = ids[i:i+L], y[i] = ids[i+L]."""
    n = len(ids) - seq_len
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    return ids[idx], ids[seq_len:seq_len + n]


def prepare_data():
    raw, tokens = load_corpus()
    split = int(len(tokens) * (1 - VAL_FRACTION))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    itos, stoi, counts = build_vocab(train_tokens)
    train_ids = encode(train_tokens, stoi)
    val_ids = encode(val_tokens, stoi)
    X_tr, y_tr = make_sequences(train_ids)
    X_va, y_va = make_sequences(val_ids)
    info = {
        "raw_characters": len(raw),
        "total_tokens": len(tokens),
        "unique_tokens_raw": len(set(tokens)),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "vocab_size": len(itos),
        "unk_rate_train": float(np.mean(train_ids == stoi["<unk>"])),
        "unk_rate_val": float(np.mean(val_ids == stoi["<unk>"])),
        "train_sequences": int(len(X_tr)),
        "val_sequences": int(len(X_va)),
        "seq_len": SEQ_LEN,
        "min_freq": MIN_FREQ,
        "top_tokens": counts.most_common(15),
    }
    return dict(itos=itos, stoi=stoi, X_tr=X_tr, y_tr=y_tr, X_va=X_va,
                y_va=y_va, info=info, tokens=tokens)


def build_model(kind, vocab_size):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inp = layers.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(vocab_size, EMBED_DIM)(inp)
    if kind == "SimpleRNN":
        x = layers.SimpleRNN(HIDDEN_UNITS)(x)
    elif kind == "BiRNN":
        x = layers.Bidirectional(layers.SimpleRNN(HIDDEN_UNITS))(x)
    elif kind == "LSTM":
        x = layers.LSTM(HIDDEN_UNITS)(x)
    elif kind == "GRU":
        x = layers.GRU(HIDDEN_UNITS)(x)
    else:
        raise ValueError(kind)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(vocab_size, activation="softmax")(x)
    model = models.Model(inp, out, name=kind)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def detokenize(tokens):
    out = []
    for t in tokens:
        if t == "\n":
            out.append("\n")
        elif t in ".,;:!?":
            if out and out[-1] != "\n":
                out[-1] = out[-1].rstrip() + t + " "
            else:
                out.append(t + " ")
        else:
            out.append(t + " ")
    return "".join(out).strip()


def save_json(obj, name):
    with open(os.path.join(RESULTS_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
