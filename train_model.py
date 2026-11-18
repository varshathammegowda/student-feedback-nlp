"""Train and evaluate the separate sentiment and category ML pipelines."""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from text_preprocessing import clean_text
from sentiment_test_cases import (SENTIMENT_TEST_CASES, CATEGORY_TEST_CASES,
                                  REPORTED_SENTIMENT_CASES)


SEED = 42
LABEL_ORDER = ["Negative", "Neutral", "Positive"]


def resolve_duplicate_labels(frame, label_col, description):
    """Keep one text per majority label; discard tied contradictory labels."""
    resolved = []
    conflicts = 0
    ties = 0
    for _, group in frame.groupby("cleaned_feedback", sort=False):
        counts = group[label_col].value_counts()
        if len(counts) > 1:
            conflicts += 1
        if len(counts) and (len(counts) == 1 or counts.iloc[0] > counts.iloc[1]):
            row = group.iloc[0].copy()
            row[label_col] = counts.index[0]
            resolved.append(row)
        else:
            ties += 1
    print(f"{description}: {conflicts} conflicting text groups; {ties} ties removed")
    return pd.DataFrame(resolved).reset_index(drop=True)


def print_metrics(name, truth, predicted, labels):
    p, r, f, _ = precision_recall_fscore_support(
        truth, predicted, labels=labels, average="macro", zero_division=0)
    print(f"\n{name}")
    print(f"Accuracy: {accuracy_score(truth, predicted):.4f}")
    print(f"Macro precision: {p:.4f}")
    print(f"Macro recall: {r:.4f}")
    print(f"Macro F1: {f:.4f}")
    print("Per-class report:\n", classification_report(
        truth, predicted, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):\n",
          confusion_matrix(truth, predicted, labels=labels))


# Main corpus supplies the original six categories.
main = pd.read_csv("dataset/student_feedback_kaggle_clean.csv")
main = main.dropna(subset=["feedback", "category", "sentiment"]).copy()
main["feedback"] = main["feedback"].astype(str)
main["cleaned_feedback"] = main["feedback"].apply(clean_text)
main = main[main["cleaned_feedback"].ne("")]
print("Main corpus rows:", len(main))
print("Main sentiment counts:\n", main["sentiment"].value_counts())
print("Original category counts:\n", main["category"].value_counts())

# Combine available sentiment sources, deduplicating by the exact normalized
# text. The repeated bare "satisfactory" labels in the source are corrected
# to Neutral per the stated annotation convention; tied conflicts are dropped.
curated = pd.read_csv("dataset/student_feedback_dataset.csv")
sentiment_extra = pd.read_csv("dataset/sentiment_training_supplement.csv")
sentiment_data = pd.concat([
    main[["feedback", "sentiment"]],
    curated[["feedback", "sentiment"]],
    sentiment_extra[["feedback", "sentiment"]],
], ignore_index=True).dropna(subset=["feedback", "sentiment"])
sentiment_data["feedback"] = sentiment_data["feedback"].astype(str)
sentiment_data["cleaned_feedback"] = sentiment_data["feedback"].apply(clean_text)
sentiment_data = sentiment_data[sentiment_data["cleaned_feedback"].ne("")]
sentiment_data.loc[sentiment_data["cleaned_feedback"].eq("satisfactory"),
                   "sentiment"] = "Neutral"
sentiment_data = resolve_duplicate_labels(
    sentiment_data, "sentiment", "Sentiment duplicate audit")
print("Final sentiment counts:\n", sentiment_data["sentiment"].value_counts())
print("Representative sentiment rows:\n", sentiment_data.groupby("sentiment", group_keys=False)
      .head(2)[["feedback", "sentiment"]].to_string(index=False, max_colwidth=90))

sent_train_text, sent_test_text, y_sent_train, y_sent_test = train_test_split(
    sentiment_data["cleaned_feedback"], sentiment_data["sentiment"],
    test_size=0.2, random_state=SEED, stratify=sentiment_data["sentiment"])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
counts = y_sent_train.value_counts()
balanced_weights = {label: len(y_sent_train) / (len(counts) * count)
                    for label, count in counts.items()}
weight_options = ["balanced"] + [
    {**balanced_weights, "Negative": balanced_weights["Negative"] * factor}
    for factor in (1.25, 1.5, 2.0, 2.5)
]
sentiment_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              min_df=1, strip_accents="unicode")),
    ("model", LogisticRegression(max_iter=3000, solver="lbfgs",
                                  random_state=SEED)),
])
sentiment_search = GridSearchCV(
    sentiment_pipeline,
    {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0],
     "model__class_weight": weight_options},
    scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
sentiment_search.fit(sent_train_text, y_sent_train)
print("\nSentiment selected parameters:", sentiment_search.best_params_)
print("Sentiment CV macro F1:", round(sentiment_search.best_score_, 4))
sent_holdout_pred = sentiment_search.best_estimator_.predict(sent_test_text)
print_metrics("Sentiment held-out evaluation", y_sent_test,
              sent_holdout_pred, LABEL_ORDER)

# Refit final sentiment model on the full deduplicated corpus. Fit TF-IDF here,
# never on uploaded feedback; its matching vectorizer is saved beside model.
sentiment_counts = sentiment_data["sentiment"].value_counts()
sent_target = sentiment_counts.max()
sent_balanced = pd.concat([
    group.sample(n=sent_target, replace=True, random_state=SEED)
    for _, group in sentiment_data.groupby("sentiment")
], ignore_index=True).sample(frac=1, random_state=SEED)
print("Balanced final sentiment counts:\n", sent_balanced["sentiment"].value_counts())
sentiment_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), sublinear_tf=True, min_df=1, strip_accents="unicode")
X_sent_balanced = sentiment_vectorizer.fit_transform(sent_balanced["cleaned_feedback"])
selected_weights = sentiment_search.best_params_["model__class_weight"]
if selected_weights == "balanced":
    deployment_weights = None
else:
    # Oversampling already removes prior class-count imbalance. Preserve only
    # the tuned relative Negative boost instead of applying raw imbalanced-data
    # weights a second time.
    negative_boost = (selected_weights["Negative"] /
                      balanced_weights["Negative"])
    deployment_weights = {"Negative": negative_boost,
                          "Neutral": 1.0, "Positive": 1.0}
print("Deployment class weights after balancing:", deployment_weights)
sentiment_model = LogisticRegression(
    C=sentiment_search.best_params_["model__C"], max_iter=3000,
    class_weight=deployment_weights,
    solver="lbfgs", random_state=SEED)
sentiment_model.fit(X_sent_balanced, sent_balanced["sentiment"])

# Exclude category texts assigned to multiple unrelated labels (for example,
# single words such as "good" and "excellent"). Add the two requested classes
# from a separate transparent training CSV; test examples are not copied there.
category_main = main[["feedback", "category", "cleaned_feedback"]].copy()
category_group_counts = category_main.groupby("cleaned_feedback")["category"].nunique()
ambiguous_category_text = set(category_group_counts[category_group_counts > 1].index)
print("Ambiguous category text groups removed:", len(ambiguous_category_text))
category_main = category_main[
    ~category_main["cleaned_feedback"].isin(ambiguous_category_text)]
category_main = category_main.drop_duplicates("cleaned_feedback")
category_extra = pd.read_csv("dataset/category_training_supplement.csv")
category_extra["cleaned_feedback"] = category_extra["feedback"].apply(clean_text)
category_data = pd.concat([
    category_main[["feedback", "category", "cleaned_feedback"]],
    category_extra[["feedback", "category", "cleaned_feedback"]],
], ignore_index=True)
category_data = category_data.drop_duplicates("cleaned_feedback")
print("Final category counts:\n", category_data["category"].value_counts())

cat_train_text, cat_test_text, y_cat_train, y_cat_test = train_test_split(
    category_data["cleaned_feedback"], category_data["category"],
    test_size=0.2, random_state=SEED, stratify=category_data["category"])
category_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
                              min_df=1, strip_accents="unicode")),
    ("model", LogisticRegression(max_iter=3000, class_weight="balanced",
                                  solver="lbfgs", random_state=SEED)),
])
category_search = GridSearchCV(
    category_pipeline, {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0]},
    scoring="f1_macro", cv=StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED), n_jobs=-1, refit=True)
category_search.fit(cat_train_text, y_cat_train)
print("\nCategory selected parameters:", category_search.best_params_)
print("Category CV macro F1:", round(category_search.best_score_, 4))
category_labels = sorted(category_data["category"].unique())
cat_holdout_pred = category_search.best_estimator_.predict(cat_test_text)
print_metrics("Category held-out evaluation", y_cat_test,
              cat_holdout_pred, category_labels)

# Persist independent model/vectorizer pairs so inference cannot mix their
# feature spaces. Refit both pairs on all validated labeled training rows.
category_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), sublinear_tf=True, min_df=1, strip_accents="unicode")
X_category = category_vectorizer.fit_transform(category_data["cleaned_feedback"])
category_model = LogisticRegression(
    C=category_search.best_params_["model__C"], max_iter=3000,
    class_weight="balanced", solver="lbfgs", random_state=SEED)
category_model.fit(X_category, category_data["category"])

print("\nSentiment label mapping:")
for i, label in enumerate(sentiment_model.classes_):
    print(f"{i} = {label}")
print("Category label mapping:")
for i, label in enumerate(category_model.classes_):
    print(f"{i} = {label}")

joblib.dump(sentiment_vectorizer, "models/sentiment_vectorizer.pkl")
joblib.dump(sentiment_model, "models/sentiment_model.pkl")
joblib.dump(category_vectorizer, "models/category_vectorizer.pkl")
joblib.dump(category_model, "models/category_model.pkl")
print("\nModel/vectorizer pairs saved.")

# Independent sentiment smoke set and expected category examples.
smoke_texts = [clean_text(text) for text, _ in SENTIMENT_TEST_CASES]
smoke_truth = [label for _, label in SENTIMENT_TEST_CASES]
smoke_pred = sentiment_model.predict(sentiment_vectorizer.transform(smoke_texts))
print_metrics("Independent sentiment smoke-set evaluation", smoke_truth,
              smoke_pred, LABEL_ORDER)
print("Sentiment smoke predictions:")
sentiment_failures = 0
for (text, expected), predicted in zip(SENTIMENT_TEST_CASES, smoke_pred):
    if expected != predicted:
        sentiment_failures += 1
        print(f"FAIL | {expected} -> {predicted} | {text}")
print(f"Smoke cases correct: {len(smoke_pred) - sentiment_failures}/{len(smoke_pred)}")
negative_cases = [(text, expected) for text, expected in SENTIMENT_TEST_CASES
                  if expected == "Negative"]
negative_pred = sentiment_model.predict(sentiment_vectorizer.transform(
    [clean_text(text) for text, _ in negative_cases]))
print(f"Negative smoke cases correct: {sum(p == 'Negative' for p in negative_pred)}/{len(negative_pred)}")

reported_texts = [clean_text(text) for text, _ in REPORTED_SENTIMENT_CASES]
reported_truth = [label for _, label in REPORTED_SENTIMENT_CASES]
reported_pred = sentiment_model.predict(
    sentiment_vectorizer.transform(reported_texts))
print("\nRequired 17-case sentiment check:")
for (text, expected), predicted in zip(REPORTED_SENTIMENT_CASES, reported_pred):
    print(f"{'PASS' if expected == predicted else 'FAIL'} | {expected} -> {predicted} | {text}")
print(f"Required cases correct: {sum(a == b for a, b in zip(reported_truth, reported_pred))}/{len(reported_truth)}")

category_texts = [clean_text(text) for text, _ in CATEGORY_TEST_CASES]
category_truth = [label for _, label in CATEGORY_TEST_CASES]
category_pred = category_model.predict(category_vectorizer.transform(category_texts))
print_metrics("Independent category-example evaluation", category_truth,
              category_pred, category_labels)
print("Category example predictions:")
category_failures = 0
for (text, expected), predicted in zip(CATEGORY_TEST_CASES, category_pred):
    if expected != predicted:
        category_failures += 1
        print(f"FAIL | {expected} -> {predicted} | {text}")
print(f"Category examples correct: {len(category_pred) - category_failures}/{len(category_pred)}")
