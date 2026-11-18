"""Shared text normalization for model training and Streamlit inference."""

import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


NEGATION_WORDS = {
    "no", "not", "nor", "never", "neither", "hardly", "scarcely", "barely",
    "cannot", "can", "couldn", "didn", "doesn", "don", "hadn", "hasn",
    "haven", "isn", "mustn", "shouldn", "wasn", "weren", "won", "wouldn",
}
SENTIMENT_WORDS = {
    "difficult", "unclear", "poor", "insufficient", "confusing", "lacking",
    "delayed", "inadequate", "disappointing", "lack", "confused", "unclear",
    "very", "too", "poorly", "without", "additional", "needed", "on", "to",
    "well", "available", "easy", "helpful", "useful", "clear", "informative",
    "engaging", "excellent", "organized", "satisfactory", "adequate",
    "manageable", "reasonable", "incomplete", "missing", "improvement", "needs",
}
PROTECTED_WORDS = NEGATION_WORDS | SENTIMENT_WORDS


def _stop_words():
    return set(stopwords.words("english")) - PROTECTED_WORDS


def clean_text(text):
    """Normalize text while retaining negation and sentiment-bearing terms."""
    text = str(text).lower().replace("\u2019", "'")
    # Expand common contractions before punctuation stripping so negation survives.
    contractions = {
        "can't": "cannot", "cannot": "cannot", "won't": "will not",
        "don't": "do not", "doesn't": "does not", "didn't": "did not",
        "isn't": "is not", "aren't": "are not", "wasn't": "was not",
        "weren't": "were not", "hasn't": "has not", "haven't": "have not",
        "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not",
        "couldn't": "could not", "mustn't": "must not",
    }
    pattern = r"\b(?:" + "|".join(re.escape(c) for c in contractions) + r")\b"
    text = re.sub(pattern, lambda m: " " + contractions[m.group(0)] + " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = word_tokenize(re.sub(r"\s+", " ", text).strip())
    stops = _stop_words()
    words = [word for word in words if word not in stops]
    lemmatizer = WordNetLemmatizer()
    return " ".join(lemmatizer.lemmatize(word) for word in words)
