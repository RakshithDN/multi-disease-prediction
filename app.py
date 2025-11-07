import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# --- Config ---
MODEL_DIR = "trained_models"

app = Flask(__name__)

# --- Load models & preprocessors (traditional only) ---
print("Loading traditional models and preprocessors...")
models = {
    "decision_tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree_model.pkl")),
    "naive_bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes_model.pkl")),
    "logistic_regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
}
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
symptom_columns = joblib.load(os.path.join(MODEL_DIR, "symptom_columns.pkl"))

print("Loaded:", list(models.keys()))
print("Number of symptoms:", len(symptom_columns))

# --- Simple text -> symptom binary vector mapper (keyword matching) ---
def text_to_symptom_vector(user_text: str, symptom_list):
    """Map user_text to binary symptom vector using substring/token matching.
       Returns vector (1/0) and list of matched symptom names."""
    text = user_text.lower()
    tokens = set([t.strip(".,!?()[]") for t in text.split()])
    vec = np.zeros(len(symptom_list), dtype=int)
    matched = []

    for i, s in enumerate(symptom_list):
        # symptom column might be like "high_fever" or "sore_throat"
        s_clean = s.replace("_", " ").lower()
        s_tokens = set(s_clean.split())

        # match rules:
        # 1) exact substring (e.g., "sore throat" in text)
        # 2) share a token (e.g., "fever" token)
        if s_clean in text or (len(tokens & s_tokens) > 0):
            vec[i] = 1
            matched.append(s)

    # fallback: if nothing matched, try to match any known token from symptoms
    if vec.sum() == 0:
        # search for the single symptom whose tokens overlap most with text tokens
        best_idx, best_score = None, 0
        for i, s in enumerate(symptom_list):
            s_clean = s.replace("_", " ").lower()
            s_tokens = set(s_clean.split())
            score = len(tokens & s_tokens)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx is not None and best_score > 0:
            vec[best_idx] = 1
            matched.append(symptom_list[best_idx])

    return vec, matched

# --- Prediction function (ensemble of 3 traditional models, soft voting) ---
def predict_disease(user_text: str, top_n: int = 3):
    vec, matched = text_to_symptom_vector(user_text, symptom_columns)
    input_df = pd.DataFrame([vec], columns=symptom_columns)
    input_scaled = scaler.transform(input_df)

    # get probability vectors
    prob_dt = models["decision_tree"].predict_proba(input_df)[0]
    prob_nb = models["naive_bayes"].predict_proba(input_df)[0]
    prob_lr = models["logistic_regression"].predict_proba(input_df)[0]

    # soft-voting average (3 models)
    avg_probs = (prob_dt + prob_nb + prob_lr) / 3.0

    top_indices = np.argsort(avg_probs)[-top_n:][::-1]
    top_diseases = le.inverse_transform(top_indices)
    top_probs = avg_probs[top_indices]

    results = {d: f"{p*100:.2f}%" for d, p in zip(top_diseases, top_probs)}
    return {"predictions": results, "matched_symptoms": matched}

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No input text provided."}), 400

    out = predict_disease(text, top_n=3)
    return jsonify(out)

# --- Run ---
if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True)
