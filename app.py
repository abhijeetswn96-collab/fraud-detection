"""
ONLINE PAYMENTS FRAUD DETECTION  -  app.py  (the Python file)
==============================================================

HOW TO RUN (VS Code):
    1. pip install -r requirements.txt
    2. Run this file (click the Run button, or type:  python app.py)
    3. Ctrl+Click the link printed in the terminal:  http://127.0.0.1:5000

PROJECT FILES (one file per language) and HOW THEY ARE CONNECTED:

    app.py  (Python)  <------ fetch() ------>  frontend/script.js  (JavaScript)
       |                                              |
       | serves the page                              | is loaded by
       v                                              v
    frontend/index.html  (HTML)  ---- <link> ---->  frontend/style.css  (CSS)

    * app.py serves the whole "frontend" folder, so the page opens at the link.
    * index.html loads style.css (<link>) and script.js (<script>).
    * script.js asks app.py for the results   -> GET  /api/summary
      and sends the payment you type to it    -> POST /predict

WHAT THIS PROGRAM DOES:
    Step 1. Load the data (Kaggle CSV, or a small fake sample if missing)
    Step 2. Prepare the data (turn text into numbers, add helpful columns)
    Step 3. Train 3 machine learning models and compare them
    Step 4. Make 3 pictures (charts)
    Step 5. Start a website where you can test a new payment
"""

# ----------------------------------------------------------------------------
# IMPORTS  (tools we use)
# ----------------------------------------------------------------------------
import base64                 # turns a picture into text so HTML can show it
import io                     # lets us save a picture in memory (not on disk)
import os                     # works with file paths
import threading              # used to open the browser after the server starts
import webbrowser             # opens the browser automatically

import matplotlib
matplotlib.use("Agg")         # draw charts without opening a window
import matplotlib.pyplot as plt
import numpy as np            # numbers / random data
import pandas as pd           # tables (called DataFrames)
from flask import Flask, jsonify, request, send_from_directory   # the website tool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------------
# Folder where this file lives
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Put the Kaggle file here with this exact name
CSV_FILE = os.path.join(BASE_FOLDER, "onlinefraud.csv")

# The real dataset has 6 million rows. We only use this many to stay fast.
MAX_ROWS = 300000

# Models understand numbers, not words. So we give each type a number.
TYPE_TO_NUMBER = {"PAYMENT": 0, "TRANSFER": 1, "CASH_OUT": 2,
                  "DEBIT": 3, "CASH_IN": 4}

# The columns (inputs) the model will learn from
FEATURE_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                   "oldbalanceDest", "newbalanceDest",
                   "errorBalanceOrig", "errorBalanceDest",
                   "origEmptied", "destIsMerchant"]


# ----------------------------------------------------------------------------
# STEP 1: LOAD THE DATA
# ----------------------------------------------------------------------------
def make_sample_data(rows=100000):
    """Make fake data with the same columns as the Kaggle file.
    Only used when onlinefraud.csv is not found."""
    rng = np.random.default_rng(42)          # random number generator

    types = rng.choice(["PAYMENT", "CASH_OUT", "CASH_IN", "TRANSFER", "DEBIT"],
                       size=rows, p=[0.34, 0.35, 0.22, 0.08, 0.01])
    amount = np.round(rng.lognormal(9.5, 1.4, rows), 2)
    old_balance = np.round(rng.lognormal(10.5, 2.0, rows), 2)

    # Fraud only happens in TRANSFER and CASH_OUT (like the real data)
    risky = np.isin(types, ["TRANSFER", "CASH_OUT"])
    is_fraud = np.zeros(rows, dtype=int)
    is_fraud[risky & (rng.random(rows) < 0.02)] = 1
    fraud_rows = np.where(is_fraud == 1)[0]

    # A fraudster usually takes ALL the money from the account
    amount[fraud_rows] = old_balance[fraud_rows]
    new_balance = np.maximum(old_balance - amount, 0)
    new_balance[types == "CASH_IN"] = (old_balance + amount)[types == "CASH_IN"]

    # Merchants have names starting with M, customers with C
    is_merchant = types == "PAYMENT"
    dest_names = np.where(is_merchant,
                          ["M" + str(n) for n in rng.integers(10**9, 2 * 10**9, rows)],
                          ["C" + str(n) for n in rng.integers(10**8, 2 * 10**9, rows)])
    old_dest = np.where(is_merchant, 0.0, np.round(rng.lognormal(10, 2, rows), 2))
    new_dest = np.where(is_merchant, 0.0, old_dest + amount)
    new_dest[fraud_rows[: len(fraud_rows) // 2]] = 0.0

    return pd.DataFrame({
        "step": rng.integers(1, 744, rows),
        "type": types,
        "amount": amount,
        "nameOrig": ["C" + str(n) for n in rng.integers(10**8, 2 * 10**9, rows)],
        "oldbalanceOrg": old_balance,
        "newbalanceOrig": np.round(new_balance, 2),
        "nameDest": dest_names,
        "oldbalanceDest": old_dest,
        "newbalanceDest": np.round(new_dest, 2),
        "isFraud": is_fraud,
    })


def load_data():
    """Read the CSV if it exists, otherwise make sample data."""
    if os.path.exists(CSV_FILE):
        print("Reading", CSV_FILE)
        data = pd.read_csv(CSV_FILE)
        source = "Kaggle file (onlinefraud.csv)"

        # If the file is huge, keep ALL fraud rows and only some normal rows
        if len(data) > MAX_ROWS:
            fraud = data[data["isFraud"] == 1]
            normal = data[data["isFraud"] == 0].sample(MAX_ROWS - len(fraud),
                                                       random_state=42)
            data = pd.concat([fraud, normal]).sample(frac=1, random_state=42)
            source += " (smaller copy of %d rows)" % len(data)
    else:
        print("onlinefraud.csv not found -> making sample data instead.")
        data = make_sample_data()
        source = "Sample data made by this program"

    return data.reset_index(drop=True), source


# ----------------------------------------------------------------------------
# STEP 2: PREPARE THE DATA
# ----------------------------------------------------------------------------
def prepare(data):
    """Turn the raw table into numbers the model can learn from."""
    d = data.copy()

    # Extra clue 1: Did the sender's balance change by exactly the amount?
    # (0 means it matches; anything else is suspicious)
    d["errorBalanceOrig"] = d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"]

    # Extra clue 2: Same check for the receiver's account
    d["errorBalanceDest"] = d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"]

    # Extra clue 3: Was the sender's account emptied completely? (1 = yes)
    d["origEmptied"] = ((d["newbalanceOrig"] == 0) & (d["oldbalanceOrg"] > 0)).astype(int)

    # Extra clue 4: Is the receiver a merchant? Names starting with "M" are.
    d["destIsMerchant"] = d["nameDest"].astype(str).str.startswith("M").astype(int)

    # Change type words (PAYMENT, TRANSFER...) into numbers
    d["type"] = d["type"].map(TYPE_TO_NUMBER)

    # Keep only the columns the model needs
    return d[FEATURE_COLUMNS]


# ----------------------------------------------------------------------------
# STEP 4 helper: turn a chart into text so the web page can show it
# ----------------------------------------------------------------------------
def chart_to_text():
    memory = io.BytesIO()
    plt.savefig(memory, format="png", bbox_inches="tight", dpi=100)
    plt.close()
    return base64.b64encode(memory.getvalue()).decode()


# ----------------------------------------------------------------------------
# STEP 3 + 4: TRAIN MODELS AND MAKE CHARTS
# ----------------------------------------------------------------------------
def train_everything():
    data, source = load_data()

    X = prepare(data)          # X = the inputs (clues)
    y = data["isFraud"]        # y = the answer (1 = fraud, 0 = normal)

    # Split: 75% to learn from (train), 25% to test on (test).
    # stratify=y keeps the same fraud percentage in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    # Our 3 models. class_weight="balanced" is important: fraud is very rare,
    # so we tell the model to pay extra attention to fraud cases.
    models = {
        # Logistic Regression likes numbers on the same scale, so we add
        # StandardScaler (it does this scaling for us).
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        # Decision Tree: asks a series of yes/no questions
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42),
        # Random Forest: many decision trees voting together
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1),
    }

    # Train each model and measure how good it is on the test data
    results = []
    for name, model in models.items():
        print("Training", name, "...")
        model.fit(X_train, y_train)               # LEARN
        guess = model.predict(X_test)             # PREDICT
        results.append({
            "name": name,
            "accuracy": accuracy_score(y_test, guess),    # how often correct
            "precision": precision_score(y_test, guess, zero_division=0),
            "recall": recall_score(y_test, guess, zero_division=0),
            "f1": f1_score(y_test, guess, zero_division=0),
        })

    # Pick the winner using F1 (a fair score when fraud is rare)
    best = max(results, key=lambda r: r["f1"])
    best_model = models[best["name"]]
    print("Best model:", best["name"])

    # ---- Make 3 charts ----
    charts = {}

    # Chart 1: transactions and fraud for each type
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    data["type"].value_counts().plot(kind="bar", ax=axes[0], title="Transactions by type")
    data.groupby("type")["isFraud"].sum().plot(
        kind="bar", ax=axes[1], color="red", title="Fraud cases by type")
    plt.tight_layout()
    charts["Data overview"] = chart_to_text()

    # Chart 2: confusion matrix = a score card of right and wrong guesses
    plt.figure(figsize=(5, 4))
    matrix = confusion_matrix(y_test, best_model.predict(X_test))
    ConfusionMatrixDisplay(matrix, display_labels=["Normal", "Fraud"]).plot(
        ax=plt.gca(), cmap="Blues", colorbar=False)
    plt.title("Confusion matrix (" + best["name"] + ")")
    charts["Confusion matrix"] = chart_to_text()

    # Chart 3: which clues mattered most (from the Random Forest)
    importance = pd.Series(models["Random Forest"].feature_importances_,
                           index=FEATURE_COLUMNS).sort_values()
    plt.figure(figsize=(5, 4))
    importance.plot(kind="barh", title="Most important clues")
    charts["Important clues"] = chart_to_text()

    # Small facts to show at the top of the page
    info = {
        "source": source,
        "total": "{:,}".format(len(data)),
        "fraud": "{:,}".format(int(y.sum())),
        "percent": "{:.3f}".format(100 * y.mean()),
        "best": best["name"],
    }
    return best_model, results, charts, info


# ----------------------------------------------------------------------------
# STEP 5: THE WEBSITE
# ----------------------------------------------------------------------------
# Folder that holds index.html, style.css and script.js
FRONTEND_FOLDER = os.path.join(BASE_FOLDER, "frontend")

# static_folder = our frontend folder, static_url_path = "" means the files are
# reachable directly:  /style.css  and  /script.js
app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")

# Train once when the program starts (this takes a few seconds)
best_model, results, charts, info = train_everything()


@app.after_request
def allow_other_pages(response):
    """Lets index.html also work if you open it with VS Code Live Server
    or by double-clicking it (a different address than this server)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def home():
    """When someone opens the link, send frontend/index.html"""
    return send_from_directory(FRONTEND_FOLDER, "index.html")


@app.route("/api/summary")
def summary():
    """script.js asks for this when the page loads.
    We send back the data facts, model scores and charts as JSON."""
    return jsonify({"info": info, "results": results, "charts": charts})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """frontend/script.js sends the form here. We answer fraud or not fraud."""
    if request.method == "OPTIONS":      # browser's "is this allowed?" check
        return ("", 204)
    try:
        form = request.get_json()

        # Put the form values into a one-row table (same columns as the CSV)
        one_payment = pd.DataFrame([{
            "step": int(form["step"]),
            "type": form["type"],
            "amount": float(form["amount"]),
            "oldbalanceOrg": float(form["oldbalanceOrg"]),
            "newbalanceOrig": float(form["newbalanceOrig"]),
            "nameDest": form["nameDest"],
            "oldbalanceDest": float(form["oldbalanceDest"]),
            "newbalanceDest": float(form["newbalanceDest"]),
        }])

        # Probability of fraud: a number from 0 (safe) to 1 (fraud)
        chance = float(best_model.predict_proba(prepare(one_payment))[0, 1])
        return jsonify({"fraud": chance >= 0.5, "chance": chance})

    except Exception as error:
        return jsonify({"error": "Please check your inputs: " + str(error)}), 400


def open_browser():
    webbrowser.open("http://127.0.0.1:5000")


# This part runs only when you run this file directly
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  APP IS READY!  Ctrl+Click this link:")
    print("  http://127.0.0.1:5000")
    print("  (Press Ctrl+C here to stop the app)")
    print("=" * 55 + "\n")
    threading.Timer(1.5, open_browser).start()   # open browser after 1.5 seconds
    app.run(host="127.0.0.1", port=5000)
