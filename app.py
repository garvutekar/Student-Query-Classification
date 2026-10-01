
from flask import Flask, request, jsonify
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

words_to_keep = {"how", "what", "why", "when", "where", "which", "can", "could",
                 "should", "would", "is", "are", "do", "does"}

try:
    stop_words = set(stopwords.words("english")) - words_to_keep
except LookupError:
    stop_words = set()

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = re.findall(r"[a-zA-Z]+", text)

    cleaned_tokens = []
    for token in tokens:
        if token in stop_words or len(token) <= 1:
            continue
        try:
            token = lemmatizer.lemmatize(token)
        except Exception:
            pass
        cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Query Classification API", "status": "running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request must contain JSON."}), 400

    if "query" not in data:
        return jsonify({"error": "JSON must contain a 'query' field."}), 400

    query = str(data["query"]).strip()

    if query == "":
        return jsonify({"error": "Query cannot be empty."}), 400

    vector = vectorizer.transform([clean_text(query)])
    prediction = model.predict(vector)
    label = str(label_encoder.inverse_transform(prediction)[0])

    response = {"query": query, "predicted_category": label}

    if hasattr(model, "predict_proba"):
        try:
            response["confidence"] = round(float(max(model.predict_proba(vector)[0])), 4)
        except Exception:
            pass

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
