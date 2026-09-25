import pandas as pd
import re
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("dataset/student_feedback_dataset.csv")

# Stopwords
stop_words = set(stopwords.words("english"))

# Keep negation words because they affect sentiment
negation_words = {
    "no", "not", "nor", "never",
    "neither", "hardly", "scarcely", "barely"
}

stop_words = stop_words - negation_words

# Lemmatizer
lemmatizer = WordNetLemmatizer()


# Text preprocessing function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
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

# Check the results
print("\nORIGINAL vs CLEANED:\n")

for i in range(5):
    print("Original :", df["feedback"].iloc[i])
    print("Cleaned  :", df["cleaned_feedback"].iloc[i])
    print("-" * 50)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert cleaned text into numerical features
X = vectorizer.fit_transform(df["cleaned_feedback"])
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("\nTF-IDF completed!")
print("Number of feedback records:", X.shape[0])
print("Number of features:", X.shape[1])

from sklearn.model_selection import train_test_split

# Target variable: feedback category
y = df["category"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split completed!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.linear_model import LogisticRegression

# Create the model
category_model = LogisticRegression(max_iter=1000)

# Train the model
category_model.fit(X_train, y_train)
joblib.dump(category_model, "models/category_model.pkl")
print("\nCategory model trained successfully!")

from sklearn.metrics import accuracy_score, classification_report

# Predict categories for test data
y_pred = category_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nCategory Classification Results")
print("Accuracy:", accuracy)

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# -----------------------------
# SENTIMENT CLASSIFICATION
# -----------------------------

# Target variable: sentiment
y_sentiment = df["sentiment"]

# Split data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X,
    y_sentiment,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment
)

# Create sentiment model
sentiment_model = LogisticRegression(max_iter=1000)

# Train model
sentiment_model.fit(X_train_s, y_train_s)

print("\nSentiment model trained successfully!")

# Make predictions
y_sentiment_pred = sentiment_model.predict(X_test_s)
joblib.dump(sentiment_model, "models/sentiment_model.pkl")
# Evaluate
sentiment_accuracy = accuracy_score(
    y_test_s,
    y_sentiment_pred
)

print("\nSentiment Classification Results")
print("Accuracy:", sentiment_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test_s,
        y_sentiment_pred
    )
)

# -----------------------------
# TEST WITH NEW FEEDBACK
# -----------------------------

new_feedback = [
    "The Wi-Fi in the hostel is extremely slow."
]

# Clean the new feedback
cleaned_new = [clean_text(text) for text in new_feedback]

# Convert to TF-IDF
new_features = vectorizer.transform(cleaned_new)

# Predict category
category_prediction = category_model.predict(new_features)

# Predict sentiment
sentiment_prediction = sentiment_model.predict(new_features)

print("\nNEW FEEDBACK TEST")
print("Feedback:", new_feedback[0])
print("Category:", category_prediction[0])
print("Sentiment:", sentiment_prediction[0])