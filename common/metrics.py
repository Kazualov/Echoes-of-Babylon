"""Shared evaluation metrics for every notebook in this project.

One implementation of corpus BLEU, corpus chrF++ and their geometric mean, used
by mBART, ByT5, Qwen and the ensemble so that every reported number is computed
the same way.

The code is a dependency-free port of sacreBLEU 2.x defaults:
  * BLEU   == sacrebleu.metrics.BLEU()                 (13a tokenizer, 'exp' smoothing)
  * chrF++ == sacrebleu.metrics.CHRF(word_order=2)     (char_order=6, beta=2)
It runs without internet (Kaggle offline scoring) and `tests/test_metrics.py`
checks that it returns exactly the same scores as sacreBLEU.

The same source is pasted into each notebook by `tools/sync_shared_cells.py`;
edit this file, then re-run that script. Do not edit the notebook copies.
"""
import math
import re
from collections import Counter

# ---------------------------------------------------------------- BLEU (13a)
_TOK_13A_RULES = [
    (re.compile(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])'), r' \1 '),
    (re.compile(r'([^0-9])([\.,])'), r'\1 \2 '),
    (re.compile(r'([\.,])([^0-9])'), r' \1 \2'),
    (re.compile(r'([0-9])(-)'), r'\1 \2 '),
]


def _tokenize_13a(line):
    line = line.replace('<skipped>', '').replace('-\n', '').replace('\n', ' ')
    if '&' in line:
        line = (line.replace('&quot;', '"').replace('&amp;', '&')
                    .replace('&lt;', '<').replace('&gt;', '>'))
    line = f' {line} '
    for pattern, repl in _TOK_13A_RULES:
        line = pattern.sub(repl, line)
    return ' '.join(line.split())


def _word_ngrams(tokens, max_order):
    return Counter(tuple(tokens[i:i + n])
                   for n in range(1, max_order + 1)
                   for i in range(len(tokens) - n + 1))


def bleu_stats(hyp, ref, max_order=4):
    """Sufficient statistics of one segment: [hyp_len, ref_len, correct_1..4, total_1..4]."""
    h = _tokenize_13a(hyp.rstrip()).split()
    r = _tokenize_13a(ref.rstrip()).split()
    h_ng, r_ng = _word_ngrams(h, max_order), _word_ngrams(r, max_order)
    correct, total = [0] * max_order, [0] * max_order
    for ng, c in h_ng.items():
        total[len(ng) - 1] += c
        if ng in r_ng:
            correct[len(ng) - 1] += min(c, r_ng[ng])
    return [len(h), len(r)] + correct + total


def bleu_from_stats(stats, max_order=4):
    sys_len, ref_len = stats[0], stats[1]
    correct, total = stats[2:2 + max_order], stats[2 + max_order:]
    bp = 1.0
    if sys_len < ref_len:
        bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0
    if not any(correct):
        return 0.0
    precisions = [0.0] * max_order
    smooth = 1.0
    for n in range(max_order):
        if total[n] == 0:
            break
        if correct[n] == 0:
            smooth *= 2
            precisions[n] = 100.0 / (smooth * total[n])
        else:
            precisions[n] = 100.0 * correct[n] / total[n]
    log_sum = sum(-9999999999 if p == 0 else math.log(p) for p in precisions)
    return bp * math.exp(log_sum / max_order)


# ---------------------------------------------------------------- chrF++
_PUNCTS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')


def _chrf_words(sent):
    out = []
    for w in sent.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


def _chrf_ngrams(sent, char_order, word_order):
    chars = ''.join(sent.split())
    grams = [Counter(chars[i:i + n] for i in range(len(chars) - n + 1))
             for n in range(1, char_order + 1)]
    words = _chrf_words(sent)
    grams += [Counter(' '.join(words[i:i + n]) for i in range(len(words) - n + 1))
              for n in range(1, word_order + 1)]
    return grams


def chrf_stats(hyp, ref, char_order=6, word_order=2):
    """Flattened [hyp, ref, match] counts for every char then word n-gram order."""
    stats = []
    for h, r in zip(_chrf_ngrams(hyp, char_order, word_order),
                    _chrf_ngrams(ref, char_order, word_order)):
        match = sum(min(c, r[g]) for g, c in h.items() if g in r)
        stats += [sum(h.values()) if r else 0, sum(r.values()), match]
    return stats


def chrf_from_stats(stats, beta=2):
    factor = beta ** 2
    avg_p = avg_r = 0.0
    eff = 0
    for i in range(len(stats) // 3):
        n_hyp, n_ref, n_match = stats[3 * i:3 * i + 3]
        if n_hyp > 0 and n_ref > 0:
            avg_p += n_match / n_hyp
            avg_r += n_match / n_ref
            eff += 1
    if eff == 0:
        return 0.0
    avg_p, avg_r = avg_p / eff, avg_r / eff
    if avg_p + avg_r == 0:
        return 0.0
    return 100.0 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


# ---------------------------------------------------------------- public API
def _sum_stats(rows):
    return [sum(col) for col in zip(*rows)] if rows else None


def segment_stats(preds, refs):
    """Per-segment (bleu_stats, chrf_stats); lets bootstrap resample without recomputing n-grams."""
    preds = ['' if p is None else str(p) for p in preds]
    refs = ['' if r is None else str(r) for r in refs]
    assert len(preds) == len(refs), f"{len(preds)} predictions vs {len(refs)} references"
    return [bleu_stats(p, r) for p, r in zip(preds, refs)], [chrf_stats(p, r) for p, r in zip(preds, refs)]


def scores_from_segment_stats(bleu_rows, chrf_rows):
    bleu = bleu_from_stats(_sum_stats(bleu_rows))
    chrfpp = chrf_from_stats(_sum_stats(chrf_rows))
    return {"bleu": bleu, "chrfpp": chrfpp, "geo_mean": math.sqrt(max(bleu, 0.0) * max(chrfpp, 0.0))}


def corpus_scores(preds, refs):
    """Corpus BLEU, chrF++ and their geometric mean (the competition score), all on 0-100."""
    return scores_from_segment_stats(*segment_stats(preds, refs))


def sentence_chrfpp(hyp, ref):
    """Sentence-level chrF++, used as the MBR utility."""
    return chrf_from_stats(chrf_stats(hyp, ref))


METRIC_SIGNATURE = "BLEU=sacrebleu-2.x-13a-exp | chrF++=sacrebleu-2.x-c6-w2-b2 | geo_mean=sqrt(BLEU*chrF++)"
