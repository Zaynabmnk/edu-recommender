"""Cleaning the video text.

The descriptions in MARS are the video subtitles joined together (Hafsa et al., 2023). When the
subtitle lines were joined, the last word of one line and the first word of the next often got stuck
together, for example "libraryto" or "Businessis". Sentences were also joined without a space after the
full stop. I repair both problems here.

To split stuck words I only use word counts from the descriptions themselves. A rare token is split
into two parts only if both parts are common words in the same corpus, so the repair cannot invent
vocabulary that is not already in the data.
"""

import ast
import re
from collections import Counter
from typing import Iterable

_NO_SPACE_AFTER_PUNCT = re.compile(r"([.,;:!?])(?=[A-Za-z])")
_TOKEN = re.compile(r"[a-z]+")
_WORD = re.compile(r"[A-Za-z]+")
_SPACES = re.compile(r"\s+")


def parse_tag_list(value) -> list:
    """Turn a stored list such as "['Excel', 'Word']" into a Python list of strings."""
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [value.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(tag).strip() for tag in parsed if str(tag).strip()]
    return [str(parsed).strip()]


def add_missing_spaces(text: str) -> str:
    """Put a space back after punctuation that runs straight into the next word."""
    return _SPACES.sub(" ", _NO_SPACE_AFTER_PUNCT.sub(r"\1 ", text)).strip()


def word_counts(texts: Iterable[str]) -> Counter:
    """Count lower-case words across all the descriptions."""
    counts = Counter()
    for text in texts:
        counts.update(_TOKEN.findall(add_missing_spaces(text).lower()))
    return counts


class GluedWordSplitter:
    """Splits tokens like "libraryto" into "library to" using corpus word counts.

    rare_max:   a token has to appear at most this many times to be treated as suspicious
    common_min: both halves have to appear at least this many times to count as real words
    min_ratio:  both halves also have to be at least this many times more common than the token,
                which stops real but uncommon words such as "workplace" from being split
    min_length: very short tokens are left alone
    """

    def __init__(self, counts: Counter, rare_max: int = 20, common_min: int = 20, min_ratio: int = 20,
                 min_length: int = 5):
        self.counts = counts
        self.rare_max = rare_max
        self.common_min = common_min
        self.min_ratio = min_ratio
        self.min_length = min_length
        self._cache = {}

    def split_token(self, token: str):
        """Return the two-word repair for a token, or None to leave it alone."""
        if token in self._cache:
            return self._cache[token]
        best = None
        count = self.counts[token]
        if len(token) >= self.min_length and count <= self.rare_max:
            needed = max(self.common_min, self.min_ratio * count)
            for cut in range(2, len(token) - 1):
                left, right = token[:cut], token[cut:]
                # I keep the split whose rarer half is most common, which
                # avoids odd splits like "t heir" when "their" was the word.
                strength = min(self.counts[left], self.counts[right])
                if strength >= needed and (best is None or strength > best[0]):
                    best = (strength, f"{left} {right}")
        repair = best[1] if best else None
        self._cache[token] = repair
        return repair

    def repair(self, text: str):
        """Clean one description. Returns the cleaned text and the number of words repaired."""
        text = add_missing_spaces(text).lower()
        repairs = 0

        def fix(match):
            nonlocal repairs
            replacement = self.split_token(match.group(0))
            if replacement is None:
                return match.group(0)
            repairs += 1
            return replacement

        return _TOKEN.sub(fix, text), repairs

    def repair_keeping_capitals(self, text: str) -> str:
        """The same repair, but keeping the original capitals, so the text reads well on the demo page."""

        def fix(match):
            word = match.group(0)
            replacement = self.split_token(word.lower())
            if replacement is None:
                return word
            cut = replacement.index(" ")
            return f"{word[:cut]} {word[cut:]}"

        return _WORD.sub(fix, add_missing_spaces(text))
