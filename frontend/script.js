/*
  script.js  -  the JavaScript file

  HOW IT IS CONNECTED:
    - index.html loads this file with <script src="script.js">
    - This file finds boxes in index.html by their id (facts, models, charts...)
    - This file talks to Python (app.py):
         GET  /api/summary  -> facts, model scores and charts (when the page opens)
         POST /predict      -> checks one payment (when you click the button)
*/

// Address of the Python server (app.py). Keep this the same as the link
// printed in the terminal.
const SERVER = "http://127.0.0.1:5000";

// ---- Find the boxes on the page (using their id names from index.html) ----
const factsBox = document.getElementById("facts");
const sourceText = document.getElementById("source");
const modelsTable = document.getElementById("models");
const chartsBox = document.getElementById("charts");
const errorBox = document.getElementById("server-error");
const form = document.getElementById("payment-form");
const resultBox = document.getElementById("result");


// ----------------------------------------------------------------------------
// PART 1: When the page opens, ask Python for the results and show them
// ----------------------------------------------------------------------------
async function loadSummary() {
    try {
        const response = await fetch(SERVER + "/api/summary");
        const data = await response.json();

        showFacts(data.info);
        showModels(data.results, data.info.best);
        showCharts(data.charts);

    } catch (problem) {
        // Python is not running (or the link is different)
        errorBox.style.display = "block";
        errorBox.textContent = "Cannot reach the Python server. " +
            "Run app.py first, then refresh this page.";
        factsBox.textContent = "";
    }
}

// Fill the 3 fact cards
function showFacts(info) {
    factsBox.innerHTML =
        makeFact(info.total, "payments used") +
        makeFact(info.fraud, "fraud payments (" + info.percent + "%)") +
        makeFact(info.best, "best model");
    sourceText.textContent = "Data source: " + info.source;
}

// Small helper: builds the HTML text of one fact card
function makeFact(big, small) {
    return '<div class="fact"><b>' + big + "</b><span>" + small + "</span></div>";
}

// Fill the table: one row for each model
function showModels(results, bestName) {
    modelsTable.innerHTML = "";
    for (const r of results) {
        const rowClass = (r.name === bestName) ? "best" : "";
        modelsTable.innerHTML +=
            '<tr class="' + rowClass + '">' +
            "<td>" + r.name + "</td>" +
            "<td>" + r.accuracy.toFixed(4) + "</td>" +
            "<td>" + r.precision.toFixed(4) + "</td>" +
            "<td>" + r.recall.toFixed(4) + "</td>" +
            "<td>" + r.f1.toFixed(4) + "</td>" +
            "</tr>";
    }
}

// Show the charts (Python sends each picture as text)
function showCharts(charts) {
    chartsBox.innerHTML = "";
    for (const title in charts) {
        chartsBox.innerHTML +=
            "<figure>" +
            '<img src="data:image/png;base64,' + charts[title] + '" alt="' + title + '">' +
            "<figcaption>" + title + "</figcaption>" +
            "</figure>";
    }
}


// ----------------------------------------------------------------------------
// PART 2: When the button is clicked, send the payment to Python
// ----------------------------------------------------------------------------
form.addEventListener("submit", async function (event) {

    // Stop the browser from reloading the page
    event.preventDefault();

    // Collect everything typed in the form into one object
    const payment = Object.fromEntries(new FormData(form).entries());

    resultBox.style.display = "block";

    try {
        // Send it to Python
        const response = await fetch(SERVER + "/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payment)
        });

        // Python replies with a small answer, e.g. {fraud: true, chance: 0.97}
        const answer = await response.json();

        // If Python sent an error message, show it in red and stop
        if (answer.error) {
            resultBox.className = "result-fraud";
            resultBox.textContent = answer.error;
            return;
        }

        // Turn 0.97 into "97.0%"
        const percent = (answer.chance * 100).toFixed(1);

        // Red box for fraud, green box for safe
        if (answer.fraud) {
            resultBox.className = "result-fraud";
            resultBox.textContent = "FRAUD suspected. Fraud chance: " + percent + "%";
        } else {
            resultBox.className = "result-safe";
            resultBox.textContent = "Looks safe. Fraud chance: " + percent + "%";
        }

    } catch (problem) {
        resultBox.className = "result-fraud";
        resultBox.textContent = "Cannot reach the Python server. Run app.py first.";
    }
});


// Start: load the results as soon as this file runs
loadSummary();
