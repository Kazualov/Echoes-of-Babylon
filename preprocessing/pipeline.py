"""
Build the sentence-level training set for the Deep Past Challenge.

Inputs:
    published_texts.csv                 full document transliterations
    Sentences_Oare_FirstWord_LinNum.csv sentence translations and first words

Outputs:
    sentence_dataset.csv   text_id, sentence_id, order, transliteration, translation, split
    extraction_report.txt  coverage, anchor success rate, diagnostics

Usage:
    python3 pipeline.py --published published_texts.csv \
        --sentences Sentences_Oare_FirstWord_LinNum.csv \
        --out sentence_dataset.csv --val-fraction 0.1 --seed 42
"""
import argparse
import csv
import random
from collections import defaultdict

from preprocessing.normalize import normalize_translation
from extract_sentences import extract_sentences_for_doc


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def build_dataset(published_rows, sentence_rows):
    """Join both sources on the document id and extract sentence pairs."""
    published_by_id = {r['oare_id'].strip(): r for r in published_rows if r.get('oare_id')}

    sentences_by_doc = defaultdict(list)
    for r in sentence_rows:
        tid = (r.get('text_uuid') or '').strip()
        if not tid:
            continue
        try:
            r = dict(r)
            r['first_word_obj_in_text'] = int(r['first_word_obj_in_text'])
        except (ValueError, TypeError):
            # Rows without a valid position cannot be used for anchor ordering.
            continue
        sentences_by_doc[tid].append(r)

    usable_ids = set(published_by_id) & set(sentences_by_doc)

    all_pairs = []
    all_failed = []
    foreign_char_log = []

    for text_id in usable_ids:
        translit_raw = published_by_id[text_id].get('transliteration', '')
        if not translit_raw:
            continue

        result = extract_sentences_for_doc(text_id, translit_raw, sentences_by_doc[text_id])
        all_failed.extend(result.failed_anchors)

        for tid, sent_id, order, span, translation in result.pairs:
            clean_translation, foreign = normalize_translation(translation)
            if foreign:
                foreign_char_log.append((tid, sent_id, foreign))
            all_pairs.append({
                'text_id': tid,
                'sentence_id': sent_id,
                'order': order,
                'transliteration': span,
                'translation': clean_translation,
            })

    stats = {
        'docs_in_published': len(published_by_id),
        'docs_in_sentences': len(sentences_by_doc),
        'docs_usable_intersection': len(usable_ids),
        'sentences_total_in_usable_docs': sum(len(sentences_by_doc[d]) for d in usable_ids),
        'pairs_extracted': len(all_pairs),
        'anchors_failed': len(all_failed),
        'rows_with_foreign_chars': len(foreign_char_log),
    }
    return all_pairs, all_failed, foreign_char_log, stats


def split_by_document(pairs, val_fraction=0.1, seed=42):
    """Split by document so all sentences from one document stay together.

    Sentence-level splitting would cause document-level data leakage and
    inflate validation scores.
    """
    doc_ids = sorted({p['text_id'] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(doc_ids)
    n_val = max(1, int(len(doc_ids) * val_fraction)) if doc_ids else 0
    val_ids = set(doc_ids[:n_val])
    for p in pairs:
        p['split'] = 'val' if p['text_id'] in val_ids else 'train'
    return pairs


def write_dataset(pairs, out_path):
    fieldnames = ['text_id', 'sentence_id', 'order', 'transliteration', 'translation', 'split']
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in pairs:
            writer.writerow(p)


def write_report(stats, all_failed, foreign_char_log, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=== Extraction stats ===\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

        if stats['sentences_total_in_usable_docs']:
            rate = 100 * (1 - stats['anchors_failed'] / stats['sentences_total_in_usable_docs'])
            f.write(f"anchor_success_rate: {rate:.1f}%\n")

        f.write("\n=== First 30 failed anchors (text_id, sentence_id, anchor, reason) ===\n")
        for row in all_failed[:30]:
            f.write(f"{row}\n")

        f.write("\n=== First 30 rows with foreign chars in translation ===\n")
        for text_id, sent_id, foreign in foreign_char_log[:30]:
            f.write(f"{text_id} / {sent_id}: {foreign}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--published', required=True, help='path to published_texts.csv')
    ap.add_argument('--sentences', required=True, help='path to Sentences_Oare_FirstWord_LinNum.csv')
    ap.add_argument('--out', default='sentence_dataset.csv')
    ap.add_argument('--report', default='extraction_report.txt')
    ap.add_argument('--val-fraction', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42, help='keep fixed for a reproducible split')
    args = ap.parse_args()

    published_rows = load_csv(args.published)
    sentence_rows = load_csv(args.sentences)

    pairs, failed, foreign_log, stats = build_dataset(published_rows, sentence_rows)
    pairs = split_by_document(pairs, args.val_fraction, args.seed)

    write_dataset(pairs, args.out)
    write_report(stats, failed, foreign_log, args.report)

    print(f"Wrote {len(pairs)} pairs to {args.out} (report: {args.report})")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
