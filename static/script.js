document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("audio", file);

    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    loading.classList.remove("hidden");
    results.innerHTML = "";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then((response) => response.json())
    .then((data) => {
        loading.classList.add("hidden");
        if (data.error) {
            results.innerHTML = `<p id="error">${data.error}</p>`;
        } else {
            const prediction = data.prediction;
            const probabilities = data.probabilities;
            results.innerHTML = `
                <div class="result">
                    <h2>Prediction: ${prediction}</h2>
                    <div class="probabilities">
                        <h3>Probabilities:</h3>
                        ${Object.entries(probabilities).map(([label, prob]) => 
                            `<p>${label}: ${(prob * 100).toFixed(2)}%</p>`
                        ).join('')}
                    </div>
                </div>
            `;
        }
    })
    .catch((error) => {
        loading.classList.add("hidden");
        results.innerHTML = `<p id="error">Error: ${error.message}</p>`;
    });
});
