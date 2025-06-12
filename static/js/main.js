document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const audioFile = document.getElementById('audioFile').files[0];
    
    if (!audioFile) {
        alert('Please select an audio file');
        return;
    }
    
    formData.append('audio', audioFile);
    
    // Show loading spinner
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const resultDiv = document.getElementById('result');
            const alertDiv = resultDiv.querySelector('.alert');
            const predictionText = document.getElementById('predictionText');
            
            // Set alert class based on prediction
            if (data.prediction === "Parkinson's Disease") {
                alertDiv.className = 'alert alert-danger';
            } else if (data.prediction === "Young Healthy Control") {
                alertDiv.className = 'alert alert-success';
            } else {
                alertDiv.className = 'alert alert-info';
            }
            
            predictionText.textContent = `Prediction: ${data.prediction}`;
            
            // Update progress bars
            const probabilities = data.probabilities;
            document.getElementById('youngHealthy').style.width = `${probabilities["Young Healthy Control"] * 100}%`;
            document.getElementById('elderlyHealthy').style.width = `${probabilities["Elderly Healthy Control"] * 100}%`;
            document.getElementById('parkinsons').style.width = `${probabilities["Parkinson's Disease"] * 100}%`;
            
            // Add percentage text
            document.getElementById('youngHealthy').textContent = `${(probabilities["Young Healthy Control"] * 100).toFixed(1)}%`;
            document.getElementById('elderlyHealthy').textContent = `${(probabilities["Elderly Healthy Control"] * 100).toFixed(1)}%`;
            document.getElementById('parkinsons').textContent = `${(probabilities["Parkinson's Disease"] * 100).toFixed(1)}%`;
            
            resultDiv.style.display = 'block';
        } else {
            throw new Error(data.error || 'An error occurred during prediction');
        }
    } catch (error) {
        alert(error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}); 