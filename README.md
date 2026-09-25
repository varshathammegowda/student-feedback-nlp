

# 🎓 Student Feedback Intelligence System

An NLP-based machine learning application that analyzes student feedback and automatically predicts its category, sentiment, important keywords, and specific issues.

## 📌 Problem Statement

Educational institutions receive a large amount of student feedback in the form of free-text comments. Manually analyzing this feedback can be time-consuming and makes it difficult to identify recurring problems.

This project uses Natural Language Processing and Machine Learning to automatically analyze student feedback.

## 🎯 Features

- Text preprocessing
- Tokenization
- Stop-word removal
- Negation preservation
- Lemmatization
- TF-IDF feature extraction
- Feedback category classification
- Sentiment classification
- TF-IDF-based keyword extraction
- Rule-based issue detection
- Interactive Streamlit dashboard

## 🧠 Machine Learning

Two Logistic Regression models are used:

### Category Classification

Predicts:

- Academics
- Administration
- Faculty
- Hostel
- Infrastructure

### Sentiment Classification

Predicts:

- Positive
- Negative
- Neutral

## 🔄 NLP Pipeline

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
TF-IDF Feature Extraction
       ↓
Machine Learning Models
       ↓
Category + Sentiment
       ↓
Keyword Extraction
       ↓
Issue Detection
````

## 📊 Dataset

The dataset contains 500 student feedback records.

Columns:

* `feedback`
* `category`
* `sentiment`

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Streamlit
* Git
* GitHub

## 📁 Project Structure

```text
student-feedback-nlp/
│
├── dataset/
│   └── student_feedback_dataset.csv
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

## ▶️ How to Run

Clone the repository:

```bash
git clone <https://github.com/varshathammegowda/student-feedback-nlp>
```

Go to the project directory:

```bash
cd student-feedback-nlp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## 🧪 Example

Input:

```text
The hostel Wi-Fi is extremely slow.
```

Output:

```text
Category: Hostel
Sentiment: Negative
Issue: Wifi
Keywords: slow, wifi, hostel
```

## 📈 Model Evaluation

The current dataset produced 1.0 accuracy on the held-out test set for both category and sentiment classification.

Because the current dataset is relatively simple, these results should not be interpreted as real-world accuracy. Testing on a larger and more diverse dataset would be required for real-world deployment.

## 🔮 Future Improvements

* Use a larger real-world student feedback dataset
* Add more sophisticated semantic models
* Improve issue detection
* Add feedback trend analysis
* Add interactive charts
* Deploy the application online
* Store feedback and predictions in a database



