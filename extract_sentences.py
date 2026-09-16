"""
Sentence-level pair extraction.

train.csv contains document-level translations, while the test set is
sentence-level. Sentences_Oare_FirstWord_LinNum.csv provides each sentence's
translation and first word. Matching these first words to the document
transliteration determines sentence boundaries and produces sentence pairs.
"""
from dataclasses import dataclass, field

from normalize import normalize_for_competition, normalize_for_matching


@dataclass
class ExtractionResult:
    """Stores extracted sentence pairs and failed anchor matches.

    pairs: (text_id, sentence_id, order, transliteration, translation)
    failed_anchors: (text_id, sentence_id, first_word_spelling, reason)
    """
    pairs: list = field(default_factory=list)
    failed_anchors: list = field(default_factory=list)


def _tokenize(text: str):
    return text.split()


def extract_sentences_for_doc(text_id: str, translit_text: str, sentence_rows: list) -> ExtractionResult:
    """Extract sentence-level pairs from one document.

    sentence_rows contains rows from Sentences_Oare_FirstWord_LinNum.csv
    for text_id. Rows are sorted by their position in the document.

    The document is normalized before storage, while anchors are normalized
    separately for matching. Search starts after the previous match to handle
    repeated words correctly. Unmatched anchors are recorded as failures.
    """
    result = ExtractionResult()

    tokens = _tokenize(normalize_for_competition(translit_text))
    rows_sorted = sorted(sentence_rows, key=lambda r: r['first_word_obj_in_text'])

    anchors = []  # (token_index, sentence_uuid, translation)
    search_from = 0

    for row in rows_sorted:
        raw_anchor = row['first_word_spelling']
        if not raw_anchor or not isinstance(raw_anchor, str):
            result.failed_anchors.append((text_id, row['sentence_uuid'], raw_anchor, 'empty_anchor'))
            continue

        anchor = normalize_for_matching(raw_anchor)

        pos = None
        for i in range(search_from, len(tokens)):
            if tokens[i] == anchor:
                pos = i
                break

        # Case-insensitive matches are accepted and recorded separately.
        if pos is None:
            for i in range(search_from, len(tokens)):
                if tokens[i].lower() == anchor.lower():
                    pos = i
                    result.failed_anchors.append(
                        (text_id, row['sentence_uuid'], raw_anchor, 'matched_case_insensitive_only')
                    )
                    break

        if pos is None:
            result.failed_anchors.append((text_id, row['sentence_uuid'], raw_anchor, 'not_found'))
            continue

        anchors.append((pos, row['sentence_uuid'], row['translation']))
        search_from = pos + 1

    # Each sentence extends from its anchor to the next sentence anchor.
    for i, (pos, sent_id, translation) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(tokens)
        result.pairs.append((text_id, sent_id, i, ' '.join(tokens[pos:end]), translation))

    return result
