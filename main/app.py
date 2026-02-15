from flask import Flask, render_template, request, jsonify, session
import pickle, numpy as np, os, re

app = Flask(__name__)
app.secret_key = "phish_sim_secret_key"

print("Loading vectorizer and models...")
vec = pickle.load(open("models/vectorizer.pkl", "rb"))
model_v = pickle.load(open("models/tableV_model.pkl", "rb"))
model_vi = pickle.load(open("models/tableVI_model.pkl", "rb"))
print("✅ Models loaded successfully!")

# static F1 scores
MODEL_F1 = {
    "Table V (SVM)": 0.844,
    "Table VI (Self-Training Ensemble)": 0.945
}

emails = [
    {
        "sender": "Amazon Payments",
        "subject": "Payment Failed — Action Required",
        "time": "10:02 AM",
        "body": """Dear Customer,

We attempted to process your order but the payment failed due to account verification issues.

Please verify your account immediately by clicking the secure link below:
http://secure-amazon.verify-login.example.com

If you do not verify within 24 hours your account will be temporarily suspended.

Regards,
Amazon Payments Team""",
        "label": "phishing"
    },
    {
        "sender": "Team Unstop",
        "subject": "Crack the Goldman Sachs Internship with this 5-day resource kit",
        "time": "6:15 PM",
        "body": """Hello B.,

We have curated a 5-day resource kit to help you prepare.

Download materials from:
https://unstop.com/resources/goldman-sachs-kit

Best of luck!""",
        "label": "legitimate"
    },
    {
        "sender": "ICICI Alerts",
        "subject": "Unusual sign-in detected — please verify",
        "time": "2:31 PM",
        "body": """Dear Customer,

We detected an unusual sign-in attempt. Please verify your identity immediately to prevent account lockout.

Verification link: http://icici-login-security.verifyuser.example.com

ICICI Bank Security Team""",
        "label": "phishing"
    }
]

@app.before_request
def init_session():
    if "score" not in session:
        session["score"] = 0
        session["attempts"] = 0

@app.route("/")
def index():
    return render_template("index.html", emails=emails, score=session["score"], attempts=session["attempts"])

def extract_reasons(text, is_phish):
    """Generate sentence-level reasoning based on keywords"""
    phishing_keywords = ["verify", "account", "login", "click", "password", "urgent", "immediately", "suspend"]
    legitimate_keywords = ["thank", "regards", "resource", "download", "information", "team"]
    sentences = re.split(r'[.!?\n]', text)
    reasons = []

    if is_phish:
        for s in sentences:
            for kw in phishing_keywords:
                if kw.lower() in s.lower():
                    reasons.append(f"⚠️ Suspicious phrase found: '{kw}' → indicates urgency or credential request.")
                    break
    else:
        for s in sentences:
            for kw in legitimate_keywords:
                if kw.lower() in s.lower():
                    reasons.append(f"✅ Legitimate indicator: '{kw}' → denotes normal professional communication.")
                    break

    if not reasons:
        reasons = ["No strong linguistic indicators found; classification based on overall context."]
    return reasons

@app.route("/verify", methods=["POST"])
def verify():
    idx = int(request.form["index"])
    mail = emails[idx]
    text = mail["body"]

    # Transform text for models
    try:
        X = vec.transform([text]).toarray()
        p_v = model_v.predict(X)[0]
        p_vi = model_vi.predict(X)[0]
    except Exception as e:
        print("⚠️ Prediction error:", e)
        p_v, p_vi = 0, 0

    avg_pred = round((p_v + p_vi) / 2)
    label_pred = "phishing" if avg_pred == 1 else "legitimate"
    true_label = mail["label"]
    correct = (label_pred == true_label)

    # update user score
    session["attempts"] += 1
    if correct:
        session["score"] += 10
        msg = "✅ Correct! You identified this email correctly."
    else:
        session["score"] = max(session["score"] - 5, 0)
        msg = f"❌ Wrong! This email was actually {true_label.upper()}."

    # reasoning
    reasons = extract_reasons(text, label_pred == "phishing")

    return jsonify({
        "predicted": label_pred,
        "true_label": true_label,
        "message": msg,
        "proof": {
            "Table V (SVM)": MODEL_F1["Table V (SVM)"],
            "Table VI (Self-Training Ensemble)": MODEL_F1["Table VI (Self-Training Ensemble)"]
        },
        "reasons": reasons,
        "score": session["score"],
        "attempts": session["attempts"]
    })

@app.route("/clicked_link", methods=["POST"])
def clicked_link():
    idx = int(request.form["index"])
    mail = emails[idx]
    session["attempts"] += 1
    if mail["label"] == "phishing":
        session["score"] = max(session["score"] - 15, 0)
        result = "⚠️ You clicked a PHISHING link! (-15 points)"
    else:
        session["score"] += 5
        result = "✅ You clicked a safe link! (+5 points)"
    return jsonify({
        "message": result,
        "score": session["score"],
        "attempts": session["attempts"]
    })

if __name__ == "__main__":
    app.run(debug=True)

