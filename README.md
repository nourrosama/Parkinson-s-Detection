# Parkinson's Detection

This project is a web application that analyzes voice recordings to detect Parkinson's disease using deep learning. Users can upload `.wav` audio files, and the app predicts whether the speaker is a Young Healthy Control, an Elderly Healthy Control, or a person with Parkinson's Disease.

---

## ✨ Features

- **Upload and Analyze**: Simple web interface to upload `.wav` files for analysis.
- **Deep Learning Model**: Uses a pre-trained TensorFlow model to classify voice samples.
- **Instant Results**: Displays prediction and confidence score.
- **Modern UI**: Responsive and clean design with HTML, CSS, and JavaScript.

---

## 🎥 Demo

![Demo Screenshot](demo-screenshot.png)

> Upload a `.wav` file and get instant predictions!

---

## 🚀 Getting Started

### ✅ Prerequisites

- Python 3.11+
- pip
- (Optional) Docker

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/nourrosama/Parkinson-s-Detection
cd Parkinson-s-Detection
```
## 2. Install dependencies

```bash
pip install -r requirements.txt
```
## 3. Download/Place the Model Files

Ensure the following files are located in the project root directory:

- `parkinson_model.h5` – Pretrained TensorFlow model
- `scaler.pkl` – Scikit-learn feature scaler
- `label_map.json` – Mapping from class indices to labels

---

## 🧪 Running the App

### Using Flask (Development)

```bash

python app.py
```
Open your browser and visit: http://localhost:5000

## 4. Usage

Once the application is running:

1. Open your browser and navigate to [http://localhost:5000](http://localhost:5000)
2. Click the **Upload** button and select a `.wav` audio file.
3. Wait a few seconds while the model processes the audio.
4. View the **prediction label** (e.g., Parkinson's Disease) and the **confidence score** returned by the model.


## 🗂️ Project Structure

project/
├── static/ # Static files (CSS, JS, images)
│ └── styles.css # Stylesheet (example)
├── templates/ # HTML templates
│ └── index.html # Main page template
├── app.py # Main Flask application script
├── parkinson_model.h5 # Pretrained TensorFlow model
├── scaler.pkl # Feature scaler for MFCC features
├── label_map.json # JSON file mapping model output to class labels
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation (this file)

## 🧠 Model & Data
Model: TensorFlow Keras model trained to classify voice samples using MFCC features.

## Labels:

Young Healthy Control

Elderly Healthy Control

Parkinson's Disease

## 🛠️ Technologies Used
Backend: Python, Flask

Machine Learning: TensorFlow, Keras

Audio Processing: Librosa

Frontend: HTML, CSS, JavaScript


