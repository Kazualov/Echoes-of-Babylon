# Deep Past Challenge — data preparation

Builds a sentence-level Akkadian→English training set for the
[Deep Past Challenge](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
and provides the text normalization used both for training data and at
inference time.

## Why this exists

The competition ships translations aligned at the **document** level
(`train.csv`), but the test set is aligned at the **sentence** level. A model
trained on whole tablets and evaluated on single sentences sees a different
input distribution at test time.

`Sentences_Oare_FirstWord_LinNum.csv` closes that gap: it contains an English
translation for every sentence, plus the first word of each sentence. Locating
those first words inside the full transliteration recovers the sentence
boundaries, producing sentence-level pairs.

Transliterations are taken from `published_texts.csv` rather than `train.csv`:
it covers 1417 of the 1700 annotated documents (`train.csv` overlaps with only
253), and its transliteration column already follows the organizers' formatting
guide.

## Files

| File | Purpose |
| --- | --- |
| `normalize.py` | Text normalization. `normalize_for_competition()` is the canonical form for model input; `normalize_for_matching()` is used internally for anchor lookup. |
| `extract_sentences.py` | Splits one document into sentences by locating each sentence's first word. |
| `pipeline.py` | CLI entry point: joins the sources, extracts pairs, assigns the split, writes the dataset and a report. |
| `test_normalize.py`, `test_extract.py` | Assertion-based tests over examples from the competition data. |

Standard library only — no third-party dependencies.

## Usage

```bash
python3 pipeline.py \
  --published published_texts.csv \
  --sentences Sentences_Oare_FirstWord_LinNum.csv \
  --out sentence_dataset.csv \
  --report extraction_report.txt \
  --val-fraction 0.1 \
  --seed 42
```

Run the tests with:

```bash
python3 test_normalize.py && python3 test_extract.py
```

### Output

`sentence_dataset.csv`:

| Column | Description |
| --- | --- |
| `text_id` | Document (tablet) identifier |
| `sentence_id` | Sentence identifier |
| `order` | Position of the sentence within its document, from 0 |
| `transliteration` | Normalized Akkadian transliteration |
| `translation` | Cleaned English translation |
| `split` | `train` or `val` |

`extraction_report.txt` reports coverage, `anchor_success_rate` (currently
99.3%), and samples of the anchors that could not be located and of unusual
characters found in translations. Both lists are diagnostic: nothing is dropped
or rewritten on their basis.

Splits are assigned per document, never per sentence, so that no document's
wording appears on both sides. Keep `--seed` fixed to reproduce a split.

## Normalization

`normalize_for_competition()` applies:

- NFC composition
- `Ḫ`/`ḫ` → `H`/`h` (the test data uses only `H`/`h`)
- Unicode subscript digits → ASCII (`il₅` → `il5`)
- `<big_gap>` → `<gap>`
- standalone `x`, `xxx`, `…` → `<gap>`
- removal of residual line dividers and half brackets, keeping enclosed text
- rounding of IEEE-754 round-trip artefacts in quantities
- whitespace collapsing

**The same function must be applied to the test transliterations before
inference.** Skipping it leaves training and test inputs in different
conventions, which degrades the score without raising an error.

```python
from normalize import normalize_for_competition

clean = normalize_for_competition(row["transliteration"])
```

`normalize_for_matching()` adds two rewrites — round-bracket determinatives to
curly ones, and `=` to `-` — that reconcile the anchor spelling in the sentence
file with the document text. Neither pattern occurs in `published_texts.csv`,
so these rules are deliberately kept out of the competition pipeline; do not
apply this function to model input.

Whenever `normalize.py` changes, rebuild `sentence_dataset.csv` and update the
copy used at inference, so that both stay on the same convention.

## Known data characteristics

- `train.csv` and `published_texts.csv` disagree on 1269 of 1561 shared
  documents. Most differences are formatting, but 86 are cases where the
  `train.csv` transliteration is truncated mid-word while its translation
  covers the full text. If document-level pairs are ever added to the training
  pool, take the transliteration from `published_texts.csv`.
- The organizers' character table lists `U+208A` for subscript x; the character
  actually present in the data is `U+2093`.
- `test.csv` as distributed is placeholder data and is not representative of
  the hidden test set — useful for checking I/O, not translation quality.
