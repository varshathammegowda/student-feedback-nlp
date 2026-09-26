# 🎓 Student Feedback Intelligence System

An NLP and Machine Learning based system that automatically analyzes student feedback, predicts feedback categories and sentiment, detects recurring issues, and provides an interactive Streamlit dashboard for batch analysis.

## 📌 Problem Statement

Educational institutions receive a large amount of student feedback in the form of textual comments.

Manually reading and analyzing this feedback can be time-consuming and makes it difficult to identify:

- Common areas of concern
- Student sentiment
- Recurring issues
- Frequently mentioned academic facilities
- Areas requiring improvement

This project uses Natural Language Processing (NLP) and Machine Learning to automatically analyze student feedback.

---

## 🎯 Project Objectives

The system aims to:

- Preprocess student feedback text
- Convert text into numerical features using TF-IDF
- Classify feedback into relevant categories
- Predict sentiment as Positive, Neutral, or Negative
- Detect specific recurring issues using rule-based detection
- Analyze large batches of feedback
- Provide filtering and downloadable results through a Streamlit dashboard

---

## 🚀 Features

### 1. Text Preprocessing

The system performs:

- Lowercase conversion
- Punctuation removal
- Tokenization
- Stop-word removal
- Negation preservation
- Lemmatization

Example:

```text
Original:
The teacher explains concepts clearly.

Processed:
teacher explains concept clearly
````

---

### 2. TF-IDF Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features.

The project uses both:

* Unigrams — individual words
* Bigrams — two-word phrases

Example:

```text
not bad
very good
not useful
```

This allows the model to consider short phrases rather than only individual words.

---

### 3. Category Classification

The system classifies feedback into six categories:

* Teaching
* Course Content
* Examination
* Lab Work
* Library Facilities
* Extra-Curricular Activities

A Logistic Regression classifier is used for category prediction.

---

### 4. Sentiment Classification

The system predicts three sentiment classes:

* Positive
* Neutral
* Negative

A separate Logistic Regression classifier is used for sentiment prediction.

Class-weight balancing is used to reduce the effect of the imbalance between sentiment classes.

---

### 5. Issue Detection

A rule-based keyword system identifies specific issues such as:

* WiFi
* Faculty
* Classroom
* Laboratory
* Hostel
* Canteen
* Examination
* Library

Example:

```text
"The hostel WiFi is extremely slow."

Detected Issue:
WiFi
```

The project therefore uses a hybrid approach:

```text
Machine Learning
    ↓
Category + Sentiment

Rule-Based Detection
    ↓
Specific Recurring Issues
```

---

## 📊 Dataset

The project uses a student feedback dataset containing feedback from students of a university in North India.

The original dataset contains six feedback categories:

1. Teaching
2. Course Content
3. Examination
4. Lab Work
5. Library Facilities
6. Extra-Curricular Activities

The original labels are:

```text
1  → Positive
0  → Neutral
-1 → Negative
```

The original dataset was reshaped into the project's working format:

```text
feedback | category | sentiment
```

After cleaning and removing unusable entries, the project contains:

**724 feedback records**

---

## 🤖 Machine Learning Pipeline

```text
Student Feedback
       ↓
Text Preprocessing
       ↓
Tokenization
       ↓
Stop-word Removal
       ↓
Lemmatization
       ↓
TF-IDF
(Unigrams + Bigrams)
       ↓
       ┌──────────────────┐
       ↓                  ↓
Category Model       Sentiment Model
       ↓                  ↓
Logistic Regression  Logistic Regression
       ↓                  ↓
Category Prediction  Sentiment Prediction
       │                  │
       └─────────┬────────┘
                 ↓
          Issue Detection
                 ↓
        Streamlit Dashboard
```

---

## 📈 Model Evaluation

The models were evaluated using an 80/20 train-test split with `random_state=42`.

### Category Classification

**Accuracy: 65.52%**

Macro F1-score:

**0.66**

### Sentiment Classification

**Accuracy: 76.55%**

Macro F1-score:

**0.70**

Sentiment classification results:

| Sentiment | Precision | Recall | F1-score |
| --------- | --------- | ------ | -------- |
| Negative  | 0.74      | 0.71   | 0.73     |
| Neutral   | 0.52      | 0.52   | 0.52     |
| Positive  | 0.85      | 0.86   | 0.85     |

The dataset contains more positive examples than neutral and negative examples. Class-weight balancing was therefore used during Logistic Regression training.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Users can upload a CSV containing a:

```text
feedback
```

column.

The system then analyzes all feedback records and displays:

* Total feedback
* Positive feedback
* Negative feedback
* Neutral feedback
* Predicted category
* Detected issue

### Filtering

Feedback can be filtered by:

* Sentiment
* Category
* Issue

### Export

Filtered results can be downloaded as a CSV file.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### NLP

* NLTK
* TF-IDF

### Machine Learning

* Scikit-learn
* Logistic Regression

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

### Dashboard

* Streamlit

### Version Control

* Git
* GitHub

---

## 📂 Project Structure

```text
student-feedback-nlp/
│
├── dataset/
│   └── student_feedback_kaggle_clean.csv
│
├── models/
│   ├── category_model.pkl
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── preprocess.py
├── train_model.py
├── keyword_extraction.py
├── issue_detection.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/varshathammegowda/student-feedback-nlp.git
```

Move into the project directory:

```bash
cd student-feedback-nlp
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the models

```bash
python train_model.py
```

### Run the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard will open in the browser.

---

## 🧪 Example

Input:

```text
The teacher explains concepts clearly but more practical examples are needed.
```

The system processes the feedback and predicts:

```text
Category: Teaching
Sentiment: Positive
```

The issue detector can additionally identify specific issues when relevant keywords are present.

---

## ⚠️ Limitations

The current system has several limitations:

* The dataset contains only 724 usable feedback records after preprocessing.
* Sentiment classes are imbalanced.
* TF-IDF and Logistic Regression have limited contextual understanding.
* Short feedback such as "Good", "Yes", or "Not bad" can be difficult to classify correctly.
* The rule-based issue detector depends on predefined keywords.
* Feedback outside the six trained categories may not be classified reliably.

Therefore, the current model results should not be interpreted as real-world accuracy for all student feedback.

---

## 🔮 Future Improvements

Possible future improvements include:

* Increasing the size and diversity of the dataset
* Adding more real student feedback
* Improving negation handling
* Using word embeddings
* Experimenting with transformer-based models such as BERT
* Improving issue detection using NLP techniques
* Adding visual analytics and charts
* Adding confidence scores for predictions
* Adding topic modeling for discovering previously unknown issues
* Deploying the Streamlit application online

---

## 👩‍💻 Author

**Varsha Thammegowda**

Engineering Student
Dayananda Sagar College of Engineering, Bengaluru


