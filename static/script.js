document.getElementById("uploadForm").addEventListener("submit", function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const loading = document.getElementById("loading");
    const results = document.getElementById("results");

    loading.classList.remove("hidden");
    results.innerHTML = "";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then((response) => response.text())
    .then((html) => {
        loading.classList.add("hidden");
        results.innerHTML = html;
    })
    .catch((error) => {
        loading.classList.add("hidden");
        results.innerHTML = `<p id="error">Error: ${error.message}</p>`;
    });
});
