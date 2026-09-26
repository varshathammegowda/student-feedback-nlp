import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# 1. Load Kaggle dataset
# --------------------------------------------------

df = pd.read_csv("dataset/student_feedback_kaggle_clean.csv")


# --------------------------------------------------
# 2. NLTK resources
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

# Keep negation words because they are important
# for sentiment analysis
negation_words = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "hardly",
    "scarcely",
    "barely"
}

# Remove negation words from the stopword list
stop_words = stop_words - negation_words

# Remove informal abbreviation "u"
stop_words.update({"u"})

lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# 3. Text preprocessing function
# --------------------------------------------------

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation/numbers with spaces
    # instead of deleting them completely
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    # but preserve negation words
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    # Convert tokens back to a sentence
    return " ".join(words)


# --------------------------------------------------
# 4. Apply preprocessing
# --------------------------------------------------

df["cleaned_feedback"] = df["feedback"].apply(clean_text)


# --------------------------------------------------
# 5. Display original vs cleaned feedback
# --------------------------------------------------

print("\nORIGINAL vs CLEANED:\n")

for i in range(min(10, len(df))):

    print("Original :", df["feedback"].iloc[i])
    print("Cleaned  :", df["cleaned_feedback"].iloc[i])
    print("-" * 60)