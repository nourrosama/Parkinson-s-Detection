import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from utils.audio_processor import process_audio

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('models/lstm_model.h5')

# Class labels
CLASS_LABELS = {
    0: "Young Healthy Control",
    1: "Elderly Healthy Control",
    2: "Parkinson's Disease"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the audio file
            processed_audio = process_audio(filepath)
            
            # Make prediction
            predictions = model.predict(processed_audio)
            predicted_class = np.argmax(predictions[0])
            probabilities = predictions[0].tolist()
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return jsonify({
                'prediction': CLASS_LABELS[predicted_class],
                'probabilities': {
                    CLASS_LABELS[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 