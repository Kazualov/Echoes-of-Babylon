"""
Сборка sentence-level (transliteration, translation) пар:
  - полный текст транслитерации документа берём из published_texts.csv
  - границы предложений + сам перевод - из Sentences_Oare_FirstWord_LinNum.csv
  - якорь - first_word_spelling, ищем последовательно по токенам документа

Обе стороны (текст документа и якоря) нормализуем ОДНИМ и тем же пайплайном
перед поиском - иначе рассинхрон конвенций (ASCII/юникод цифры) даёт ложные
"якорь не найден".
"""
from dataclasses import dataclass, field
from normalize import normalize_transliteration


@dataclass
class ExtractionResult:
    pairs: list = field(default_factory=list)     # [(text_id, sentence_id, order, transliteration, translation)]
    failed_anchors: list = field(default_factory=list)  # [(text_id, sentence_id, first_word_spelling, reason)]


def _tokenize(text: str):
    return text.split()


def extract_sentences_for_doc(text_id: str, translit_text: str, sentence_rows: list) -> ExtractionResult:
    """
    sentence_rows: список dict с ключами
        sentence_uuid, sentence_obj_in_text, translation, first_word_spelling, first_word_obj_in_text
    для ОДНОГО документа (text_id), ещё не отсортированный.
    """
    result = ExtractionResult()

    norm_text = normalize_transliteration(translit_text)
    tokens = _tokenize(norm_text)

    rows_sorted = sorted(sentence_rows, key=lambda r: r['first_word_obj_in_text'])

    anchors = []  # (token_index, sentence_uuid, translation)
    search_from = 0
    for row in rows_sorted:
        raw_anchor = row['first_word_spelling']
        if not raw_anchor or not isinstance(raw_anchor, str):
            result.failed_anchors.append((text_id, row['sentence_uuid'], raw_anchor, 'empty_anchor'))
            continue

        anchor = normalize_transliteration(raw_anchor)

        pos = None
        for i in range(search_from, len(tokens)):
            if tokens[i] == anchor:
                pos = i
                break

        if pos is None:
            # fallback: попробовать без учёта регистра (caps несут смысл, но
            # лучше найти предложение с чуть менее строгим совпадением, чем
            # потерять его совсем - логируем как "fuzzy", а не молча принимаем)
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

    for i, (pos, sent_id, translation) in enumerate(anchors):
        end = anchors[i + 1][0] if i + 1 < len(anchors) else len(tokens)
        span = ' '.join(tokens[pos:end])
        result.pairs.append((text_id, sent_id, i, span, translation))

    return result
