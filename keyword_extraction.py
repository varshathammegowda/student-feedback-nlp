import joblib
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
def extract_keywords(text):
    vector = vectorizer.transform([text])

    feature_names = vectorizer.get_feature_names_out()

    scores = vector.toarray()[0]

    word_scores = list(zip(feature_names, scores))

    word_scores.sort(key=lambda x: x[1], reverse=True)

    keywords = [word for word, score in word_scores if score > 0][:5]

    return keywords

text = "hostel wifi slow library"

keywords = extract_keywords(text)

print("Input:", text)
print("Keywords:", keywords)