"""Builds README.md; every number in the tables is read from results/*.json."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c

R = c.RESULTS_DIR
info = json.load(open(os.path.join(R, "dataset_info.json")))
met = json.load(open(os.path.join(R, "metrics.json")))
gen = json.load(open(os.path.join(R, "generation_metrics.json")))
samples = json.load(open(os.path.join(R, "generated_samples.json")))

KINDS = c.MODEL_NAMES
LABEL = {"SimpleRNN": "Simple RNN", "BiRNN": "Bidirectional RNN", "LSTM": "LSTM", "GRU": "GRU"}


def eval_table():
    rows = ["| Model | Parameters | Epochs run (best) | Train loss | Val loss | Val perplexity | Val accuracy | Val top-5 accuracy | Training time (s) | Time / epoch (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in KINDS:
        m = met[k]
        rows.append(f"| {LABEL[k]} | {m['parameters']:,} | {m['epochs_run']} ({m['best_epoch']}) | "
                    f"{m['train_loss']:.3f} | {m['val_loss']:.3f} | {m['val_perplexity']:.2f} | "
                    f"{m['val_accuracy']*100:.1f}% | {m['val_top5_accuracy']*100:.1f}% | "
                    f"{m['train_time_sec']:.0f} | {m['time_per_epoch_sec']:.0f} |")
    return "\n".join(rows)


def temp_table():
    rows = ["| Temperature | Model | Distinct-1 | Distinct-2 | Repeated-trigram ratio | Immediate-repeat ratio | Corpus-bigram validity |",
            "|---:|---|---:|---:|---:|---:|---:|"]
    for T in c.TEMPERATURES:
        for k in KINDS:
            g = gen[k][str(T)]
            rows.append(f"| {T} | {LABEL[k]} | {g['distinct1']:.3f} | {g['distinct2']:.3f} | "
                        f"{g['repeated_trigram_ratio']:.3f} | {g['immediate_repeat_ratio']:.3f} | "
                        f"{g['bigram_validity']:.3f} |")
    return "\n".join(rows)


def sample_block(kind, T, seed="1"):
    txt = samples[kind][str(T)][seed][0]
    return "\n".join("    " + line for line in txt.split("\n"))


top = ", ".join(f"`{t if t != chr(10) else chr(92) + 'n'}` ({n:,})" for t, n in info["top_tokens"][:8])

README = f"""# Autoregressive Word-Level Text Generation with Simple RNN, Bidirectional RNN, LSTM and GRU

Course: Foundations of Generative AI (ICT4403), Assignment 2

## 1. Problem Statement and Objective

The task is to build an autoregressive text-generation system that predicts the next word of a text from the preceding words and then generates text one word at a time. Four recurrent architectures are implemented: Simple RNN, Bidirectional RNN, LSTM and GRU. All four are trained on the same prepared data under the same experimental conditions. The objectives are to:

1. describe and preprocess a textual dataset and build the input-target sequences for autoregressive modelling;
2. compare the models quantitatively (loss, perplexity, accuracy, training time) and qualitatively (coherence, diversity, contextual consistency, repetition);
3. generate text from identical seed sequences at three sampling temperatures, analyse the effect of temperature, and justify the most effective model for the dataset.

## 2. Dataset

| Item | Value |
|---|---|
| Name | Tiny Shakespeare |
| Content | Concatenated text of Shakespeare's plays (dialogue with speaker labels) |
| Source link | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt |
| Repository page | https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare |
| Local copy | `data/input.txt` |
| Size | {info['raw_characters']:,} characters, {info['total_tokens']:,} tokens after tokenisation |
| Unique tokens (raw) | {info['unique_tokens_raw']:,} |

**Characteristics.** The text is written in Early Modern English with an archaic lexicon, verse line breaks and speaker labels followed by a colon (for example `first citizen:`). The most frequent tokens are {top}. Line breaks and punctuation carry structural information (speaker changes, verse lines), so they are kept as tokens.

**Preprocessing.**

1. Convert to lower case.
2. Tokenise into words (apostrophes inside a word are kept, e.g. `we'll`), the punctuation marks `. , ; : ! ?`, and the newline character as separate tokens.
3. Split chronologically: the first 90% of tokens form the training set ({info['train_tokens']:,} tokens) and the last 10% the validation set ({info['val_tokens']:,} tokens). The validation text comes from different passages, so there is no window overlap with training data.
4. Build the vocabulary from the training split only, keeping words with frequency of at least {info['min_freq']}; all other words map to `<unk>`. Vocabulary size is {info['vocab_size']:,} including `<pad>` and `<unk>`. The `<unk>` rate is {info['unk_rate_train']*100:.1f}% in training and {info['unk_rate_val']*100:.1f}% in validation.
5. Create input-target pairs with a sliding window of stride 1 and length {info['seq_len']}: the input is words *i* to *i+{info['seq_len']-1}* and the target is word *i+{info['seq_len']}*. This gives {info['train_sequences']:,} training and {info['val_sequences']:,} validation sequences.

## 3. Models Implemented

All models share the same structure: `Embedding({c.EMBED_DIM})` followed by one recurrent layer with {c.HIDDEN_UNITS} units per direction, `Dropout({c.DROPOUT})`, and a `Dense(vocabulary size, softmax)` output layer.

| Model | Recurrent layer | Note |
|---|---|---|
| Simple RNN | `SimpleRNN({c.HIDDEN_UNITS})` | Plain recurrence with tanh; prone to vanishing gradients |
| Bidirectional RNN | `Bidirectional(SimpleRNN({c.HIDDEN_UNITS}))` | Reads the {info['seq_len']}-word input window forwards and backwards; the target word is never visible. Output is 2 x {c.HIDDEN_UNITS} units, so it has more parameters |
| LSTM | `LSTM({c.HIDDEN_UNITS})` | Input, forget and output gates with a cell state |
| GRU | `GRU({c.HIDDEN_UNITS})` | Update and reset gates; fewer gates than LSTM |

**Common training conditions.** Loss: sparse categorical cross-entropy. Optimiser: Adam, learning rate {c.LEARNING_RATE}. Batch size {c.BATCH_SIZE}. Maximum {c.EPOCHS} epochs with early stopping on validation loss (patience 2, best weights restored). Random seed {c.SEED} for every model. Training ran on CPU (2 cores).

## 4. Execution Steps and Dependencies

Dependencies are listed in `requirements.txt` (developed and run with Python 3.13 and TensorFlow 2.21 / Keras 3; TensorFlow/Keras, NumPy and Matplotlib are required).

```bash
pip install -r requirements.txt
mkdir -p data
curl -L -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt   # skip if data/input.txt exists
python train.py        # trains the four models, writes models/ and results/metrics.json (about 40 minutes on 2 CPU cores)
python generate.py     # generates text at three temperatures, writes results/generated_samples.* and generation_metrics.json
python make_plots.py   # writes results/plots/*.png
python build_readme.py # regenerates the tables in this README from the JSON logs
```

| File | Purpose |
|---|---|
| `common.py` | Configuration, preprocessing, sequence construction, model definitions |
| `train.py` | Training and evaluation of the four models |
| `generate.py` | Temperature sampling and generation metrics |
| `make_plots.py` | Plots |
| `results/` | Metrics (JSON), training log, generated samples, plots |
| `models/` | Trained Keras models |

## 5. Results

### 5.1 Quantitative evaluation

Perplexity is the exponential of the cross-entropy loss. Train metrics were computed after training with dropout disabled. Validation text is a different part of the corpus than training text, which is why validation accuracy can be slightly above training accuracy for some models.

{eval_table()}

![Training curves](results/plots/training_curves.png)

![Model comparison](results/plots/model_comparison.png)

### 5.2 Generated text

Each model was given the same three seed passages (the last {info['seq_len']} tokens of each are used as context) and generated {c.GEN_WORDS} words. Seeds 1 and 3 (the "first citizen" opening and the Romeo balcony line) occur in the training text; seed 2 (Hamlet, "to be, or not to be") does not. Special tokens `<pad>` and `<unk>` are excluded from sampling. Full output for every model, seed and temperature (8 samples each) is in `results/generated_samples.json`; the first sample of each is in `results/generated_samples.txt`.

Seed 1, temperature 0.5, GRU:

{sample_block('GRU', 0.5)}

Seed 1, temperature 1.0, GRU:

{sample_block('GRU', 1.0)}

Seed 1, temperature 1.5, GRU:

{sample_block('GRU', 1.5)}

## 6. Model Comparison and Evaluation

**Prediction quality.** The GRU is the best model on every predictive measure: validation loss {met['GRU']['val_loss']:.3f}, perplexity {met['GRU']['val_perplexity']:.1f}, accuracy {met['GRU']['val_accuracy']*100:.1f}% and top-5 accuracy {met['GRU']['val_top5_accuracy']*100:.1f}%. The Simple RNN ({met['SimpleRNN']['val_perplexity']:.1f}) and LSTM ({met['LSTM']['val_perplexity']:.1f}) are close to each other, and the Bidirectional RNN is the weakest ({met['BiRNN']['val_perplexity']:.1f}). The GRU advantage over the next best model is about {met['SimpleRNN']['val_perplexity']-met['GRU']['val_perplexity']:.1f} perplexity points; the other three are within about {met['BiRNN']['val_perplexity']-met['SimpleRNN']['val_perplexity']:.0f} points of each other.

**Convergence and overfitting.** The Bidirectional RNN has the most parameters ({met['BiRNN']['parameters']:,}) and its validation loss stopped improving after epoch {met['BiRNN']['best_epoch']}, while its training loss kept falling, so it overfits first. The Simple RNN also plateaued (best epoch {met['SimpleRNN']['best_epoch']}). The LSTM converged more slowly than the others and its best epoch was the last one, as was the GRU's; both were still improving when the {c.EPOCHS}-epoch cap was reached, so their final numbers are not converged values.

**Training time.** Simple RNN is the cheapest ({met['SimpleRNN']['train_time_sec']:.0f} s), the Bidirectional RNN costs about 10% more in total ({met['BiRNN']['train_time_sec']:.0f} s) despite stopping earlier, the LSTM about twice as much ({met['LSTM']['train_time_sec']:.0f} s) and the GRU the most ({met['GRU']['train_time_sec']:.0f} s). The GRU being slower than the LSTM is a property of the CPU implementation used here, not of the architecture, and the timings come from a shared 2-core machine, so they should be read as relative rather than absolute.

**Qualitative characteristics.** All four models learn the surface form of the play: speaker labels followed by a colon and a line break, short verse-like lines, period vocabulary ("my lord", "thou", character names). None produces globally coherent meaning at this scale; sentences are locally plausible and then drift. Contextual consistency is weak for all models: in the samples read, the speaker chosen after the seed is usually unrelated to it, and lines rarely respond to the preceding speaker. At T = 0.5 the GRU samples read slightly more grammatical in the samples reviewed (for example "the gods have done, sir, to this good senator"), but only a small number of samples were read and this difference was not measured, so it is an impression rather than a result.

## 7. Sampling-Temperature Analysis

Sampling divides the log-probabilities by the temperature T before the softmax. Three values were used: 0.5, 1.0 and 1.5. Each cell below averages 3 seeds x {8} samples x {c.GEN_WORDS} words. Metrics:

- **Distinct-1 / Distinct-2:** fraction of unique words / unique word pairs (diversity; higher is more varied).
- **Repeated-trigram ratio:** share of word triples that repeat within a sample (repetition; lower is better).
- **Immediate-repeat ratio:** share of adjacent identical words.
- **Corpus-bigram validity:** share of generated word pairs that occur in the training text. This is only a rough coherence proxy: a pair can exist in the corpus and still be nonsensical in context, and a fluent but novel pair is counted as invalid.

{temp_table()}

![Temperature effects](results/plots/temperature_effects.png)

**T = 0.5 (conservative).** Output is the most fluent: {min(gen[k]['0.5']['bigram_validity'] for k in KINDS)*100:.0f}% to {max(gen[k]['0.5']['bigram_validity'] for k in KINDS)*100:.0f}% of word pairs are valid corpus pairs for all four models, and the text uses common frames. The cost is low diversity (distinct-2 about {gen['SimpleRNN']['0.5']['distinct2']:.2f} to {gen['GRU']['0.5']['distinct2']:.2f}) and the highest repetition ({gen['LSTM']['0.5']['repeated_trigram_ratio']*100:.1f}% to {gen['SimpleRNN']['0.5']['repeated_trigram_ratio']*100:.1f}% repeated trigrams). Reading the samples, the repetition is also thematic: generic phrases such as "is a man" and "the king" recur across a passage, which the trigram metric underestimates.

**T = 1.0 (unmodified distribution).** Repetition essentially disappears (repeated-trigram ratio at or below {max(gen[k]['1.0']['repeated_trigram_ratio'] for k in KINDS):.3f}) and diversity is high (distinct-2 about 0.99). Corpus-bigram validity falls to {min(gen[k]['1.0']['bigram_validity'] for k in KINDS):.2f} to {max(gen[k]['1.0']['bigram_validity'] for k in KINDS):.2f}. Text keeps the play format and readable phrases but contains frequent semantic slips and ungrammatical joins.

**T = 1.5 (flattened).** Diversity saturates (distinct-2 of about 1.0) but validity drops to {min(gen[k]['1.5']['bigram_validity'] for k in KINDS):.2f} to {max(gen[k]['1.5']['bigram_validity'] for k in KINDS):.2f}. The output is a sequence of rare words with little syntax, and speaker structure is lost.

**Overall effect.** Raising the temperature trades coherence for diversity. Within the range tested, repetition is only a problem at T = 0.5 and coherence is only acceptable at T = 0.5 to 1.0, so a temperature between 0.5 and 1.0 is the likely best compromise. Values between those two were not tested. The trend is the same for all four models; the differences between models at a given temperature are small compared with the differences between temperatures.

## 8. Conclusion

The **GRU** is the best-performing model for this dataset. It reaches the lowest validation perplexity ({met['GRU']['val_perplexity']:.1f} against {met['SimpleRNN']['val_perplexity']:.1f}, {met['LSTM']['val_perplexity']:.1f} and {met['BiRNN']['val_perplexity']:.1f}) and the highest next-word accuracy, and at T = 1.0 it has the highest diversity ({gen['GRU']['1.0']['distinct2']:.3f} distinct-2), the highest corpus-bigram validity ({gen['GRU']['1.0']['bigram_validity']:.3f}) and the lowest immediate-repeat ratio among the four. The Bidirectional RNN gave no benefit for next-word prediction (the information needed is mostly in the most recent words) and overfitted earliest. The Simple RNN is a strong, cheap baseline. The price of the GRU result is the longest training time in this setup.

### Limitations

- Each model was trained once (seed {c.SEED}); differences between the Simple RNN, LSTM and Bidirectional RNN (a few perplexity points) may lie within run-to-run variation, and no confidence intervals were computed. The GRU margin is larger but also from a single run.
- The LSTM and GRU were still improving at the {c.EPOCHS}-epoch cap; with more training the ranking could change, in particular for the LSTM.
- Coherence and contextual consistency were judged by reading samples and a bigram-based proxy, not by human or model-based evaluation.
- The vocabulary is limited to words with frequency of at least {info['min_freq']}, so rare words (many proper names) cannot be generated.
- Seeds 1 and 3 occur in the training data, so those generations may partly reflect memorised text; seed 2 does not.
"""

with open(os.path.join(c.ROOT, "README.md"), "w", encoding="utf-8") as f:
    f.write(README)
print("README.md written", len(README), "chars")
