import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("dataset/student_feedback_dataset.csv")

# NLTK resources
stop_words = set(stopwords.words("english"))

# Keep negation words because they affect sentiment
negation_words = {
    "no", "not", "nor", "never",
    "neither", "hardly", "scarcely", "barely"
}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords but keep negation words
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


# Apply preprocessing
df["cleaned_feedback"] = df["feedback"].apply(clean_text)

# Display original and cleaned text
print("\nORIGINAL vs CLEANED:\n")

for i in range(10):
    print("Original :", df["feedback"].iloc[i])
    print("Cleaned  :", df["cleaned_feedback"].iloc[i])
    print("-" * 60)