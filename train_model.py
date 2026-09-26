import pandas as pd
import joblib
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(
    "dataset/student_feedback_kaggle_clean.csv"
)

print("Dataset loaded successfully!")
print("Total feedback records:", len(df))


# --------------------------------------------------
# 2. Text preprocessing
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

# Remove negation words from stopwords
stop_words = stop_words - negation_words

# Remove informal abbreviation
stop_words.update({"u"})

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    # Convert to string
    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace punctuation/numbers with spaces
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = word_tokenize(text)

    # Stopword removal
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

    return " ".join(words)


# Apply preprocessing
df["cleaned_feedback"] = (
    df["feedback"].apply(clean_text)
)

print("\nText preprocessing completed!")


# --------------------------------------------------
# 3. Prepare input and labels
# --------------------------------------------------

X_text = df["cleaned_feedback"]

y_category = df["category"]

y_sentiment = df["sentiment"]


# --------------------------------------------------
# 4. TF-IDF Vectorization
# --------------------------------------------------

# Learn both individual words and word pairs.
#
# (1,1) = only individual words
# (1,2) = individual words + two-word phrases
#
# This helps with phrases such as:
# "not bad"
# "not good"
# "very good"
# "not useful"

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1
)

X = vectorizer.fit_transform(X_text)

print("\nTF-IDF completed!")

print(
    "Number of feedback records:",
    X.shape[0]
)

print(
    "Number of TF-IDF features:",
    X.shape[1]
)


# --------------------------------------------------
# 5. Category train-test split
# --------------------------------------------------

X_train_cat, X_test_cat, y_category_train, y_category_test = (
    train_test_split(
        X,
        y_category,
        test_size=0.2,
        random_state=42,
        stratify=y_category
    )
)

print("\nCategory training samples:", X_train_cat.shape[0])
print("Category testing samples:", X_test_cat.shape[0])


# --------------------------------------------------
# 6. Category Classification Model
# --------------------------------------------------

category_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

category_model.fit(
    X_train_cat,
    y_category_train
)

category_predictions = category_model.predict(
    X_test_cat
)

category_accuracy = accuracy_score(
    y_category_test,
    category_predictions
)

print("\nCategory model trained successfully!")

print("\nCategory Classification Results")

print(
    "Accuracy:",
    category_accuracy
)

print("\nClassification Report:")

print(
    classification_report(
        y_category_test,
        category_predictions
    )
)


# --------------------------------------------------
# 7. Sentiment train-test split
# --------------------------------------------------

X_train_sent, X_test_sent, y_sentiment_train, y_sentiment_test = (
    train_test_split(
        X,
        y_sentiment,
        test_size=0.2,
        random_state=42,
        stratify=y_sentiment
    )
)

print(
    "\nSentiment training samples:",
    X_train_sent.shape[0]
)

print(
    "Sentiment testing samples:",
    X_test_sent.shape[0]
)


# --------------------------------------------------
# 8. Sentiment Classification Model
# --------------------------------------------------

sentiment_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

sentiment_model.fit(
    X_train_sent,
    y_sentiment_train
)

sentiment_predictions = sentiment_model.predict(
    X_test_sent
)

sentiment_accuracy = accuracy_score(
    y_sentiment_test,
    sentiment_predictions
)

print("\nSentiment model trained successfully!")

print("\nSentiment Classification Results")

print(
    "Accuracy:",
    sentiment_accuracy
)

print("\nClassification Report:")

print(
    classification_report(
        y_sentiment_test,
        sentiment_predictions
    )
)


# --------------------------------------------------
# 9. Save trained models
# --------------------------------------------------

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    category_model,
    "models/category_model.pkl"
)

joblib.dump(
    sentiment_model,
    "models/sentiment_model.pkl"
)

print("\nModels saved successfully!")


# --------------------------------------------------
# 10. Test new feedback
# --------------------------------------------------

test_feedback = [
    "Not bad",
    "The teacher explains concepts clearly but more practical examples are needed.",
    "The examination was very difficult and poorly organized."
]

print("\nNEW FEEDBACK TESTS")

for text in test_feedback:

    cleaned = clean_text(text)

    vector = vectorizer.transform(
        [cleaned]
    )

    predicted_category = (
        category_model.predict(vector)[0]
    )

    predicted_sentiment = (
        sentiment_model.predict(vector)[0]
    )

    print("\nFeedback:", text)
    print("Cleaned :", cleaned)
    print("Category:", predicted_category)
    print("Sentiment:", predicted_sentiment)