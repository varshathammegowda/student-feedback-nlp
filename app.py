import streamlit as st
import joblib
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Feedback Intelligence",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title(" Student Feedback Intelligence System")

st.write(
    "AI-powered analysis of student feedback using NLP and Machine Learning."
)


# ============================================================
# NLTK SETUP
# ============================================================

stop_words = set(stopwords.words("english"))

# Keep negation words because they affect sentiment
negation_words = {
    "no", "not", "nor", "never",
    "neither", "hardly", "scarcely", "barely"
}

stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords but preserve negation words
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


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

category_model = joblib.load(
    "models/category_model.pkl"
)

sentiment_model = joblib.load(
    "models/sentiment_model.pkl"
)


# ============================================================
# ISSUE DETECTION
# ============================================================

issues = {
    "wifi": ["wifi", "internet", "network", "connection"],
    "faculty": ["faculty", "professor", "teacher", "lecturer"],
    "classroom": ["classroom", "class", "lecture hall"],
    "laboratory": ["laboratory", "lab", "equipment"],
    "hostel": ["hostel", "room", "accommodation"],
    "canteen": ["canteen", "food", "mess"],
    "examination": ["exam", "examination", "test", "assessment"],
    "library": ["library", "books", "reading"],
}


def detect_issue(text):

    text = text.lower()

    for issue, keywords in issues.items():

        for keyword in keywords:

            if keyword in text:
                return issue

    return "Other"


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text):

    vector = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()

    scores = vector.toarray()[0]

    word_scores = list(
        zip(feature_names, scores)
    )

    # Highest TF-IDF score first
    word_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    keywords = [
        word
        for word, score in word_scores
        if score > 0
    ][:5]

    return keywords


# ============================================================
# SUCCESS MESSAGE
# ============================================================




# ============================================================
# FEEDBACK INPUT
# ============================================================

feedback = st.text_area(
    "Enter student feedback:",
    placeholder="Example: The hostel Wi-Fi is very slow."
)


# ============================================================
# ANALYZE FEEDBACK
# ============================================================

if st.button(" Analyze Feedback"):

    if feedback.strip():

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        cleaned_feedback = clean_text(feedback)


        # ----------------------------------------------------
        # TF-IDF TRANSFORMATION
        # ----------------------------------------------------

        features = vectorizer.transform(
            [cleaned_feedback]
        )


        # ----------------------------------------------------
        # CATEGORY PREDICTION
        # ----------------------------------------------------

        category_prediction = category_model.predict(
            features
        )[0]


        # ----------------------------------------------------
        # SENTIMENT PREDICTION
        # ----------------------------------------------------

        sentiment_prediction = sentiment_model.predict(
            features
        )[0]


        # ----------------------------------------------------
        # KEYWORD EXTRACTION
        # ----------------------------------------------------

        keywords = extract_keywords(
            cleaned_feedback
        )


        # ----------------------------------------------------
        # ISSUE DETECTION
        # ----------------------------------------------------

        issue = detect_issue(
            cleaned_feedback
        )


        # ====================================================
        # DISPLAY RESULTS
        # ====================================================

        st.subheader(" Analysis Results")


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Category",
                category_prediction
            )


        with col2:

            st.metric(
                "Sentiment",
                sentiment_prediction
            )


        st.subheader(" Detected Issue")

        st.info(issue.title())


        st.subheader("Important Keywords")

        if keywords:

            st.write(
                ", ".join(keywords)
            )

        else:

            st.write(
                "No keywords detected."
            )


        st.subheader("Processed Feedback")

        st.code(
            cleaned_feedback
        )


    else:

        st.warning(
            "Please enter some feedback."
        )