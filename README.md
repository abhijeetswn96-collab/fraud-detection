# Online Payments Fraud Detection with Machine Learning

A beginner-friendly project: a machine learning model learns which online
payments are fraud, and a small website lets you test new payments.

## Folder structure (one file per language)
```
fraud_connected/
|-- app.py                  Python     : load data, train models, run server
|-- requirements.txt        packages to install
|-- onlinefraud.csv         (optional) the Kaggle dataset - put it here
`-- frontend/
    |-- index.html          HTML       : page structure
    |-- style.css           CSS        : colours and layout
    `-- script.js           JavaScript : talks to Python, fills the page
```

## How the files are connected
```
 app.py (Python) <---- fetch() ----> script.js (JavaScript)
     |                                    |
     | serves the page                    | loaded by <script src="script.js">
     v                                    v
 index.html (HTML) ---- <link href="style.css"> ----> style.css (CSS)
```
| From | To | How |
|---|---|---|
| app.py | index.html, style.css, script.js | serves the `frontend` folder at the link |
| index.html | style.css | `<link rel="stylesheet" href="style.css">` |
| index.html | script.js | `<script src="script.js"></script>` |
| script.js | index.html | finds boxes by id (`facts`, `models`, `charts`, `result`) |
| script.js | app.py | `GET /api/summary` (results) and `POST /predict` (check a payment) |

## How to run in VS Code
1. Open the `fraud_connected` folder in VS Code (File > Open Folder).
2. Open the terminal (Ctrl + `) and install packages once:
   ```
   pip install -r requirements.txt
   ```
3. Run `app.py` (Run button, or `python app.py`).
4. Ctrl + Click the link in the terminal: **http://127.0.0.1:5000**

(Optional) You can also open `frontend/index.html` with VS Code Live Server
or by double-clicking it - it will still find the Python server, as long as
`app.py` is running.

## Using the real dataset
Download the Kaggle "Online Payments Fraud Detection Dataset", rename it to
`onlinefraud.csv`, and put it next to `app.py`. If the file is missing, the
program makes sample data so it still works.

## How it works (in simple words)
1. **Load data** - read the payments table.
2. **Prepare data** - turn words into numbers and add clues, e.g. "was the
   sender's account emptied?".
3. **Train** - 3 models learn from 75% of the data: Logistic Regression,
   Decision Tree, Random Forest.
4. **Test** - check them on the other 25% they have never seen. Fraud is very
   rare, so we judge with Precision, Recall and F1 (not only Accuracy).
5. **Website** - the best model answers: fraud or not?
