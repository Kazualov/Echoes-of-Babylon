"""
Text normalization for the Deep Past Challenge (Old Assyrian Akkadian -> English).

Two distinct normalization pipelines are exposed; they must not be confused:

    normalize_for_competition(text)
        Canonical form of a transliteration as fed to the model. Used both when
        building the training set and at inference time on the hidden test set.
        This is the only function that needs to be shipped to the Kaggle kernel.

    normalize_for_matching(text)
        normalize_for_competition() plus two extra rules that exist solely to
        reconcile the anchor format of Sentences_Oare_FirstWord_LinNum.csv with
        the document text. For internal use by extract_sentences.py only; never
        apply it to model input.

Rules follow the organizers' Dataset Instructions where those are explicit, and
are otherwise derived from measurements on the full competition files. Counts
quoted in the comments refer to those measurements.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Competition normalization
# ---------------------------------------------------------------------------

# Test data uses H/h instead of Ḫ/ḫ; normalize to H/h.
_HA_MAP = str.maketrans({'Ḫ': 'H', 'ḫ': 'h'})


def normalize_ha(text: str) -> str:
    """Fold Ḫ/ḫ to H/h."""
    return text.translate(_HA_MAP)


# Convert Unicode subscript digits to ASCII digits (e.g. il₅ -> il5).
_SUBSCRIPT_TO_ASCII_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def subscript_to_ascii_digit(text: str) -> str:
    """Convert Unicode subscript digits to ASCII digits."""
    return text.translate(_SUBSCRIPT_TO_ASCII_MAP)


def unify_gaps(text: str) -> str:
    """Normalize <big_gap> to <gap>."""
    return text.replace('<big_gap>', '<gap>')


# Replace standalone x/… tokens with <gap>; keep x inside valid syllables.
_BROKEN_TOKEN = re.compile(r'^[xX]+$')


def normalize_broken_tokens(text: str) -> str:
    """Replace whole tokens standing for an illegible sign with <gap>."""
    tokens = text.split()
    return ' '.join(
        '<gap>' if (t == '…' or _BROKEN_TOKEN.match(t)) else t
        for t in tokens
    )


# Residual scribal notation that survived the organizers' own cleaning:
#   '/'      line divider, 8 occurrences in published_texts.csv
#   '⌈' '⌉'  half brackets (partially broken sign), 1 occurrence in each file
# Every observed '/' (9 of 9) directly follows a hyphen, i.e. a tablet line
# break that coincides with a syllable boundary; the hyphen already separates
# the syllables, so the slash is dropped. The pattern requires the hyphen rather
# than deleting every '/', so an unseen standalone slash is left untouched
# instead of being guessed at.
_HALF_BRACKETS = str.maketrans('', '', '˹˺⌈⌉')
_SLASH_AFTER_HYPHEN = re.compile(r'-/')


def strip_scribal_marks(text: str) -> str:
    """Drop line dividers and half brackets, keeping the enclosed text."""
    text = _SLASH_AFTER_HYPHEN.sub('-', text)
    return text.translate(_HALF_BRACKETS)


# Round decimal artifacts with more than five fractional digits.
_FLOAT_ARTIFACT = re.compile(r'\d+\.\d{6,}')


def round_float_artifacts(text: str, ndigits: int = 5) -> str:
    """Round decimals with an implausible number of places back to ndigits."""
    return _FLOAT_ARTIFACT.sub(lambda m: f"{float(m.group(0)):.{ndigits}f}", text)


def normalize_for_competition(text: str) -> str:
    """Canonical transliteration form for model input (training and inference)."""
    if text is None:
        return text
    text = unicodedata.normalize('NFC', text)
    text = normalize_ha(text)
    text = subscript_to_ascii_digit(text)
    text = unify_gaps(text)
    text = normalize_broken_tokens(text)
    text = strip_scribal_marks(text)
    text = round_float_artifacts(text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Anchor matching (extract_sentences.py only)
# ---------------------------------------------------------------------------

# Sentence anchors use (X) for determinatives; documents use {X}.
# This rule is used only for anchor matching.
_PAREN_DETERMINATIVE = re.compile(r'\(([^()]+)\)')


def normalize_parenthetical_determinatives(text: str) -> str:
    """Rewrite (X) determinatives as {X}."""
    return _PAREN_DETERMINATIVE.sub(r'{\1}', text)


def normalize_ligature_equals(text: str) -> str:
    """Replace the '=' lemma boundary in anchors with '-'."""
    return text.replace('=', '-')


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching sentence anchors to document text."""
    if text is None:
        return text
    text = normalize_for_competition(text)
    text = normalize_parenthetical_determinatives(text)
    text = normalize_ligature_equals(text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Translation normalization
# ---------------------------------------------------------------------------

# Allowed Unicode ranges cover observed Latin characters, diacritics,
# fractions, quotation marks, dashes, and subscript characters.
# U+00B4 is excluded and reported for manual review.
_ALLOWED_RANGES = [
    (0x0000, 0x007F),   # ASCII
    (0x00A0, 0x00B3),   # Latin-1 Supplement, up to ACUTE ACCENT
    (0x00B5, 0x024F),   # rest of Latin-1 Supplement + Latin Extended-A/B (½ ¾ à š ṣ ṭ)
    (0x02B0, 0x02FF),   # Spacing Modifier Letters (ʾ U+02BE)
    (0x1E00, 0x1EFF),   # Latin Extended Additional (Ṣ, Ṭ, Ḫ)
    (0x2013, 0x2014),   # en/em dash
    (0x2018, 0x201F),   # typographic quotes
    (0x2026, 0x2026),   # ellipsis
    (0x2044, 0x2044),   # fraction slash
    (0x2080, 0x2089),   # subscript digits (e.g. Puzur₄-Aššur)
    (0x2090, 0x209C),   # subscript modifier letters, incl. U+2093 ₓ
    (0x2150, 0x218F),   # Number Forms (⅚ and similar)
]


def _is_allowed(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ALLOWED_RANGES)


def scan_foreign_chars(text: str):
    """Return characters outside the allowed Unicode ranges."""
    found = []
    if text is None:
        return found
    for i, ch in enumerate(text):
        if not _is_allowed(ch):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = '<unknown>'
            found.append((i, ch, name))
    return found


def balance_quotes(text: str) -> str:
    """Remove anomalous trailing OCR quotes while preserving valid quotes."""
    if text is None:
        return text
    t = text.strip()
    t = re.sub(
        r'["\u201c\u201d\u201e\s]+$',
        lambda m: '' if m.group(0).count('"') != 1 else m.group(0),
        t,
    )
    return t.strip()


def normalize_translation(text: str):
    """Clean a translation and return detected foreign characters."""
    if text is None:
        return text, []
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\xa0', ' ')
    foreign = scan_foreign_chars(text)
    cleaned = balance_quotes(text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned, foreign
