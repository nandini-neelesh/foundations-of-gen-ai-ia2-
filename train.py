"""Train Simple RNN, Bidirectional RNN, LSTM and GRU next-word predictors
under identical conditions and log loss / perplexity / accuracy / time."""
import os
import sys
import time
import json
import math

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
import common as c


def main():
    data = c.prepare_data()
    c.save_json(data["info"], "dataset_info.json")
    with open(os.path.join(c.RESULTS_DIR, "vocab.json"), "w") as f:
        json.dump(data["itos"], f)

    V = data["info"]["vocab_size"]
    all_results = {}
    histories = {}

    for kind in c.MODEL_NAMES:
        print(f"\n===== Training {kind} =====", flush=True)
        c.set_seeds()
        model = c.build_model(kind, V)
        n_params = model.count_params()

        stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True)

        t0 = time.time()
        hist = model.fit(
            data["X_tr"], data["y_tr"],
            validation_data=(data["X_va"], data["y_va"]),
            batch_size=c.BATCH_SIZE, epochs=c.EPOCHS,
            callbacks=[stop], verbose=2, shuffle=True)
        train_time = time.time() - t0

        epochs_run = len(hist.history["loss"])
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

        tr_loss, tr_acc = model.evaluate(data["X_tr"], data["y_tr"],
                                         batch_size=1024, verbose=0)
        va_loss, va_acc = model.evaluate(data["X_va"], data["y_va"],
                                         batch_size=1024, verbose=0)
        # top-5 accuracy on validation set
        probs = model.predict(data["X_va"], batch_size=1024, verbose=0)
        top5 = np.argsort(-probs, axis=1)[:, :5]
        va_top5 = float(np.mean([y in row for y, row in zip(data["y_va"], top5)]))

        all_results[kind] = {
            "parameters": int(n_params),
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "train_time_sec": round(train_time, 1),
            "time_per_epoch_sec": round(train_time / epochs_run, 1),
            "train_loss": float(tr_loss),
            "train_perplexity": float(math.exp(tr_loss)),
            "train_accuracy": float(tr_acc),
            "val_loss": float(va_loss),
            "val_perplexity": float(math.exp(va_loss)),
            "val_accuracy": float(va_acc),
            "val_top5_accuracy": va_top5,
        }
        histories[kind] = {k: [float(v) for v in vals]
                           for k, vals in hist.history.items()}
        model.save(os.path.join(c.MODELS_DIR, f"{kind}.keras"))
        c.save_json(all_results, "metrics.json")
        c.save_json(histories, "histories.json")
        print(json.dumps(all_results[kind], indent=1), flush=True)

    print("\nALL TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
