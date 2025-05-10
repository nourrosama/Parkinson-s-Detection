from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav'}

model = tf.keras.models.load_model("parkinson_model.h5")

label_map = {
    0: 'Young Healthy Control',
    1: 'Elderly Healthy Control',
    2: "Parkinson's Disease"
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_mfcc(file_path, max_len=200, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc[..., np.newaxis]  


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            mfcc = extract_mfcc(filepath)
            mfcc = mfcc[np.newaxis, ...]  # batch dimension

            prediction = model.predict(mfcc)
            predicted_label = np.argmax(prediction)
            result = label_map[predicted_label]
            confidence = float(np.max(prediction))

            return f"Prediction: {result}<br>Confidence: {confidence * 100:.2f}%"

        except Exception as e:
            return f"Error: {str(e)}", 500

    return "Invalid file type. Please upload a .wav file.", 400

if __name__ == '__main__':
    app.run(debug=True)
