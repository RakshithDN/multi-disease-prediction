const btn = document.getElementById("predictBtn");
const input = document.getElementById("symptomInput");
const status = document.getElementById("status");
const resultsBox = document.getElementById("results");
const predList = document.getElementById("predList");
const matchedDiv = document.getElementById("matched");

btn.addEventListener("click", async () => {
    const text = input.value.trim();
    predList.innerHTML = "";
    matchedDiv.innerHTML = "";
    resultsBox.classList.add("hidden");

    if (!text) {
        status.textContent = "Please enter some symptoms.";
        return;
    }

    status.textContent = "Analyzing symptoms…";
    btn.disabled = true;

    try {
        const resp = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        if (!resp.ok) {
            const err = await resp.json();
            status.textContent = err.error || "Server error";
            btn.disabled = false;
            return;
        }

        const data = await resp.json();
        status.textContent = "";

        // show predictions
        const preds = data.predictions || {};
        predList.innerHTML = "";
        Object.entries(preds).forEach(([disease, prob]) => {
            const p = document.createElement("p");
            p.innerHTML = `${disease} — <span style="font-weight:700">${prob}</span>`;
            predList.appendChild(p);
        });

        // matched symptoms
        const matched = data.matched_symptoms || [];
        if (matched.length === 0) {
            matchedDiv.innerHTML = "<span style='color:#9aa4b2'>No explicit symptom matched</span>";
        } else {
            matchedDiv.innerHTML = "";
            matched.forEach(s => {
                const sp = document.createElement("span");
                sp.textContent = s.replace(/_/g, " ");
                matchedDiv.appendChild(sp);
            });
        }

        resultsBox.classList.remove("hidden");
    } catch (e) {
        console.error(e);
        status.textContent = "Error connecting to server.";
    } finally {
        btn.disabled = false;
    }
});
