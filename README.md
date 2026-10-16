# Autoregressive Word-Level Text Generation with Simple RNN, Bidirectional RNN, LSTM and GRU

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
| Size | 1,115,394 characters, 290,148 tokens after tokenisation |
| Unique tokens (raw) | 12,380 |

**Characteristics.** The text is written in Early Modern English with an archaic lexicon, verse line breaks and speaker labels followed by a colon (for example `first citizen:`). The most frequent tokens are `\n` (35,662), `,` (17,798), `:` (9,081), `.` (6,991), `the` (5,739), `and` (5,087), `to` (4,473), `i` (4,137). Line breaks and punctuation carry structural information (speaker changes, verse lines), so they are kept as tokens.

**Preprocessing.**

1. Convert to lower case.
2. Tokenise into words (apostrophes inside a word are kept, e.g. `we'll`), the punctuation marks `. , ; : ! ?`, and the newline character as separate tokens.
3. Split chronologically: the first 90% of tokens form the training set (261,133 tokens) and the last 10% the validation set (29,015 tokens). The validation text comes from different passages, so there is no window overlap with training data.
4. Build the vocabulary from the training split only, keeping words with frequency of at least 5; all other words map to `<unk>`. Vocabulary size is 3,101 including `<pad>` and `<unk>`. The `<unk>` rate is 5.4% in training and 8.5% in validation.
5. Create input-target pairs with a sliding window of stride 1 and length 20: the input is words *i* to *i+19* and the target is word *i+20*. This gives 261,113 training and 28,995 validation sequences.

## 3. Models Implemented

All models share the same structure: `Embedding(64)` followed by one recurrent layer with 128 units per direction, `Dropout(0.2)`, and a `Dense(vocabulary size, softmax)` output layer.

| Model | Recurrent layer | Note |
|---|---|---|
| Simple RNN | `SimpleRNN(128)` | Plain recurrence with tanh; prone to vanishing gradients |
| Bidirectional RNN | `Bidirectional(SimpleRNN(128))` | Reads the 20-word input window forwards and backwards; the target word is never visible. Output is 2 x 128 units, so it has more parameters |
| LSTM | `LSTM(128)` | Input, forget and output gates with a cell state |
| GRU | `GRU(128)` | Update and reset gates; fewer gates than LSTM |

**Common training conditions.** Loss: sparse categorical cross-entropy. Optimiser: Adam, learning rate 0.001. Batch size 256. Maximum 10 epochs with early stopping on validation loss (patience 2, best weights restored). Random seed 42 for every model. Training ran on CPU (2 cores).

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

| Model | Parameters | Epochs run (best) | Train loss | Val loss | Val perplexity | Val accuracy | Val top-5 accuracy | Training time (s) | Time / epoch (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple RNN | 623,197 | 10 (8) | 3.713 | 3.949 | 51.90 | 30.3% | 48.8% | 370 | 37 |
| Bidirectional RNN | 1,044,829 | 7 (5) | 3.804 | 4.007 | 54.98 | 29.8% | 47.8% | 405 | 58 |
| LSTM | 697,309 | 10 (10) | 3.835 | 3.962 | 52.58 | 30.0% | 48.9% | 749 | 75 |
| GRU | 672,989 | 10 (10) | 3.501 | 3.852 | 47.07 | 31.1% | 50.4% | 942 | 94 |

![Training curves](results/plots/training_curves.png)

![Model comparison](results/plots/model_comparison.png)

### 5.2 Generated text

Each model was given the same three seed passages (the last 20 tokens of each are used as context) and generated 60 words. Seeds 1 and 3 (the "first citizen" opening and the Romeo balcony line) occur in the training text; seed 2 (Hamlet, "to be, or not to be") does not. Special tokens `<pad>` and `<unk>` are excluded from sampling. Full output for every model, seed and temperature (8 samples each) is in `results/generated_samples.json`; the first sample of each is in `results/generated_samples.txt`.

Seed 1, temperature 0.5, GRU:

    in the way of honour. 
    
    sicinius: 
    they are not: 
    he is a man, for he is the city. 
    
    duke vincentio: 
    the gods have done, sir, to this good senator. 
    
    duke vincentio: 
    and you shall be your resolution,

Seed 1, temperature 1.0, GRU:

    , 
    i will live at the remembrance of your love. 
    
    gloucester: 
    i'll meet him from my love, hold up. 
    
    montague: 
    
    juliet: 
    romeo!, we say to the king should grace my crown. 
    
    richmond: 
    tis gone to set my leave

Seed 1, temperature 1.5, GRU:

    bear 
    honesty twas home about that royal gracious wills: 
    those half his keeper go forced leontes, die: 
    but ne'er told course or princely self bodies doth 
    in romeo; so dare i thrive money were thy heaven reigns. nest 
    in any hour unto a war that edge who speaks,

## 6. Model Comparison and Evaluation

**Prediction quality.** The GRU is the best model on every predictive measure: validation loss 3.852, perplexity 47.1, accuracy 31.1% and top-5 accuracy 50.4%. The Simple RNN (51.9) and LSTM (52.6) are close to each other, and the Bidirectional RNN is the weakest (55.0). The GRU advantage over the next best model is about 4.8 perplexity points; the other three are within about 3 points of each other.

**Convergence and overfitting.** The Bidirectional RNN has the most parameters (1,044,829) and its validation loss stopped improving after epoch 5, while its training loss kept falling, so it overfits first. The Simple RNN also plateaued (best epoch 8). The LSTM converged more slowly than the others and its best epoch was the last one, as was the GRU's; both were still improving when the 10-epoch cap was reached, so their final numbers are not converged values.

**Training time.** Simple RNN is the cheapest (370 s), the Bidirectional RNN costs about 10% more in total (405 s) despite stopping earlier, the LSTM about twice as much (749 s) and the GRU the most (942 s). The GRU being slower than the LSTM is a property of the CPU implementation used here, not of the architecture, and the timings come from a shared 2-core machine, so they should be read as relative rather than absolute.

**Qualitative characteristics.** All four models learn the surface form of the play: speaker labels followed by a colon and a line break, short verse-like lines, period vocabulary ("my lord", "thou", character names). None produces globally coherent meaning at this scale; sentences are locally plausible and then drift. Contextual consistency is weak for all models: in the samples read, the speaker chosen after the seed is usually unrelated to it, and lines rarely respond to the preceding speaker. At T = 0.5 the GRU samples read slightly more grammatical in the samples reviewed (for example "the gods have done, sir, to this good senator"), but only a small number of samples were read and this difference was not measured, so it is an impression rather than a result.

## 7. Sampling-Temperature Analysis

Sampling divides the log-probabilities by the temperature T before the softmax. Three values were used: 0.5, 1.0 and 1.5. Each cell below averages 3 seeds x 8 samples x 60 words. Metrics:

- **Distinct-1 / Distinct-2:** fraction of unique words / unique word pairs (diversity; higher is more varied).
- **Repeated-trigram ratio:** share of word triples that repeat within a sample (repetition; lower is better).
- **Immediate-repeat ratio:** share of adjacent identical words.
- **Corpus-bigram validity:** share of generated word pairs that occur in the training text. This is only a rough coherence proxy: a pair can exist in the corpus and still be nonsensical in context, and a fluent but novel pair is counted as invalid.

| Temperature | Model | Distinct-1 | Distinct-2 | Repeated-trigram ratio | Immediate-repeat ratio | Corpus-bigram validity |
|---:|---|---:|---:|---:|---:|---:|
| 0.5 | Simple RNN | 0.626 | 0.900 | 0.024 | 0.003 | 0.972 |
| 0.5 | Bidirectional RNN | 0.621 | 0.902 | 0.022 | 0.005 | 0.960 |
| 0.5 | LSTM | 0.634 | 0.924 | 0.016 | 0.002 | 0.957 |
| 0.5 | GRU | 0.655 | 0.929 | 0.020 | 0.002 | 0.968 |
| 1.0 | Simple RNN | 0.796 | 0.994 | 0.000 | 0.004 | 0.737 |
| 1.0 | Bidirectional RNN | 0.794 | 0.993 | 0.000 | 0.003 | 0.729 |
| 1.0 | LSTM | 0.788 | 0.990 | 0.001 | 0.002 | 0.711 |
| 1.0 | GRU | 0.803 | 0.997 | 0.000 | 0.001 | 0.747 |
| 1.5 | Simple RNN | 0.896 | 0.999 | 0.000 | 0.002 | 0.393 |
| 1.5 | Bidirectional RNN | 0.892 | 1.000 | 0.000 | 0.002 | 0.394 |
| 1.5 | LSTM | 0.889 | 1.000 | 0.000 | 0.001 | 0.408 |
| 1.5 | GRU | 0.890 | 0.999 | 0.000 | 0.001 | 0.416 |

![Temperature effects](results/plots/temperature_effects.png)

**T = 0.5 (conservative).** Output is the most fluent: 96% to 97% of word pairs are valid corpus pairs for all four models, and the text uses common frames. The cost is low diversity (distinct-2 about 0.90 to 0.93) and the highest repetition (1.6% to 2.4% repeated trigrams). Reading the samples, the repetition is also thematic: generic phrases such as "is a man" and "the king" recur across a passage, which the trigram metric underestimates.

**T = 1.0 (unmodified distribution).** Repetition essentially disappears (repeated-trigram ratio at or below 0.001) and diversity is high (distinct-2 about 0.99). Corpus-bigram validity falls to 0.71 to 0.75. Text keeps the play format and readable phrases but contains frequent semantic slips and ungrammatical joins.

**T = 1.5 (flattened).** Diversity saturates (distinct-2 of about 1.0) but validity drops to 0.39 to 0.42. The output is a sequence of rare words with little syntax, and speaker structure is lost.

**Overall effect.** Raising the temperature trades coherence for diversity. Within the range tested, repetition is only a problem at T = 0.5 and coherence is only acceptable at T = 0.5 to 1.0, so a temperature between 0.5 and 1.0 is the likely best compromise. Values between those two were not tested. The trend is the same for all four models; the differences between models at a given temperature are small compared with the differences between temperatures.

## 8. Conclusion

The **GRU** is the best-performing model for this dataset. It reaches the lowest validation perplexity (47.1 against 51.9, 52.6 and 55.0) and the highest next-word accuracy, and at T = 1.0 it has the highest diversity (0.997 distinct-2), the highest corpus-bigram validity (0.747) and the lowest immediate-repeat ratio among the four. The Bidirectional RNN gave no benefit for next-word prediction (the information needed is mostly in the most recent words) and overfitted earliest. The Simple RNN is a strong, cheap baseline. The price of the GRU result is the longest training time in this setup.

### Limitations

- Each model was trained once (seed 42); differences between the Simple RNN, LSTM and Bidirectional RNN (a few perplexity points) may lie within run-to-run variation, and no confidence intervals were computed. The GRU margin is larger but also from a single run.
- The LSTM and GRU were still improving at the 10-epoch cap; with more training the ranking could change, in particular for the LSTM.
- Coherence and contextual consistency were judged by reading samples and a bigram-based proxy, not by human or model-based evaluation.
- The vocabulary is limited to words with frequency of at least 5, so rare words (many proper names) cannot be generated.
- Seeds 1 and 3 occur in the training data, so those generations may partly reflect memorised text; seed 2 does not.
