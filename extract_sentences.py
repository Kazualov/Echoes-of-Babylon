"""
Извлечение sentence-level пар по якорному слову из Sentences_Oare_FirstWord_LinNum.csv

  - полный текст транслитерации документа нормализуется normalize_for_competition() -
    это тот же вид, что попадёт в sentence_dataset.csv и что будет применяться к
    hidden test на инференсе. Именно ЭТОТ нормализованный текст режется на спаны
    и сохраняется как есть.
  - якорь (first_word_spelling) нормализуется normalize_for_matching() - расширенным
    набором правил (+ круглые скобки, + '='), которые нужны только для того, чтобы
    формат анкора из Sentences-файла совпал с форматом документа. Эти доп. правила
    - no-op на самом документе (0 вхождений '(' и '=' в published_texts.csv,
    проверено эмпирически), поэтому расхождение нормализации между двумя сторонами
    не создаёт скрытого искажения текста документа.
"""
from dataclasses import dataclass, field
from normalize import normalize_for_competition, normalize_for_matching


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

    norm_text = normalize_for_competition(translit_text)
    tokens = _tokenize(norm_text)

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
