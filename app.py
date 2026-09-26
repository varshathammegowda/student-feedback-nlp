import streamlit as st
import pandas as pd
import joblib
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# 1. Streamlit configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Feedback Intelligence",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Feedback Intelligence System")

st.write(
    "Analyze student feedback using NLP, TF-IDF, "
    "machine learning, and issue detection."
)


# --------------------------------------------------
# 2. Text preprocessing
# --------------------------------------------------

stop_words = set(stopwords.words("english"))

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

# Preserve negation words
stop_words = stop_words - negation_words

# Remove informal abbreviation
stop_words.update({"u"})

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Replace punctuation and numbers with spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

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


# --------------------------------------------------
# 3. Load trained models
# --------------------------------------------------

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

category_model = joblib.load(
    "models/category_model.pkl"
)

sentiment_model = joblib.load(
    "models/sentiment_model.pkl"
)


# --------------------------------------------------
# 4. Issue detection
# --------------------------------------------------

issues = {

    "Wifi": [
        "wifi",
        "internet",
        "network",
        "connection"
    ],

    "Faculty": [
        "faculty",
        "professor",
        "teacher",
        "lecturer"
    ],

    "Classroom": [
        "classroom",
        "class",
        "lecture hall"
    ],

    "Laboratory": [
        "laboratory",
        "lab",
        "equipment"
    ],

    "Hostel": [
        "hostel",
        "room",
        "accommodation"
    ],

    "Canteen": [
        "canteen",
        "food",
        "mess"
    ],

    "Examination": [
        "exam",
        "examination",
        "test",
        "assessment"
    ],

    "Library": [
        "library",
        "books",
        "reading"
    ]
}


def detect_issue(text):

    text = text.lower()

    for issue, keywords in issues.items():

        for keyword in keywords:

            if keyword in text:
                return issue

    return "Other"


# --------------------------------------------------
# 5. Upload feedback dataset
# --------------------------------------------------

st.subheader("📂 Upload Student Feedback")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing a 'feedback' column",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # --------------------------------------------------
    # Validate dataset
    # --------------------------------------------------

    if "feedback" not in df.columns:

        st.error(
            "The CSV must contain a column named 'feedback'."
        )

    else:

        st.success(
            f"Successfully uploaded {len(df)} feedback records."
        )


        # --------------------------------------------------
        # 6. Preprocess feedback
        # --------------------------------------------------

        df["cleaned_feedback"] = (
            df["feedback"].apply(clean_text)
        )


        # --------------------------------------------------
        # 7. Convert text to TF-IDF
        # --------------------------------------------------

        features = vectorizer.transform(
            df["cleaned_feedback"]
        )


        # --------------------------------------------------
        # 8. Predict category
        # --------------------------------------------------

        df["category"] = (
            category_model.predict(features)
        )


        # --------------------------------------------------
        # 9. Predict sentiment
        # --------------------------------------------------

        df["sentiment"] = (
            sentiment_model.predict(features)
        )


        # --------------------------------------------------
        # 10. Detect specific issue
        # --------------------------------------------------

        df["issue"] = (
            df["feedback"].apply(detect_issue)
        )


        # --------------------------------------------------
        # 11. Summary
        # --------------------------------------------------

        st.subheader("📊 Feedback Summary")

        total = len(df)

        positive = (
            df["sentiment"] == "Positive"
        ).sum()

        negative = (
            df["sentiment"] == "Negative"
        ).sum()

        neutral = (
            df["sentiment"] == "Neutral"
        ).sum()


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Total Feedback",
                total
            )


        with col2:

            st.metric(
                "Positive",
                positive
            )


        with col3:

            st.metric(
                "Negative",
                negative
            )


        with col4:

            st.metric(
                "Neutral",
                neutral
            )


        # --------------------------------------------------
        # 12. Filters
        # --------------------------------------------------

        st.subheader("🔎 Filter Feedback")

        col1, col2, col3 = st.columns(3)


        with col1:

            sentiment_filter = st.selectbox(
                "Sentiment",
                ["All"] +
                sorted(
                    df["sentiment"].unique()
                )
            )


        with col2:

            category_filter = st.selectbox(
                "Category",
                ["All"] +
                sorted(
                    df["category"].unique()
                )
            )


        with col3:

            issue_filter = st.selectbox(
                "Issue",
                ["All"] +
                sorted(
                    df["issue"].unique()
                )
            )


        # --------------------------------------------------
        # 13. Apply filters
        # --------------------------------------------------

        filtered_df = df.copy()


        if sentiment_filter != "All":

            filtered_df = filtered_df[
                filtered_df["sentiment"]
                == sentiment_filter
            ]


        if category_filter != "All":

            filtered_df = filtered_df[
                filtered_df["category"]
                == category_filter
            ]


        if issue_filter != "All":

            filtered_df = filtered_df[
                filtered_df["issue"]
                == issue_filter
            ]


        st.write(
            f"Showing {len(filtered_df)} "
            f"of {len(df)} feedback records."
        )


        # --------------------------------------------------
        # 14. Display results
        # --------------------------------------------------

        st.subheader("📋 Feedback Results")

        display_columns = [
            "feedback",
            "category",
            "sentiment",
            "issue"
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True
        )


        # --------------------------------------------------
        # 15. Download results
        # --------------------------------------------------

        csv_data = filtered_df[
            display_columns
        ].to_csv(index=False)


        st.download_button(
            label="📥 Download Filtered Results",

            data=csv_data,

            file_name="analyzed_student_feedback.csv",

            mime="text/csv"
        )