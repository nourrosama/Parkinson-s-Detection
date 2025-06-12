import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def visualize_feature(feature, title, sr=22050):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, x_axis='time', sr=sr)
    plt.colorbar(format="%+2.f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def extract_mfcc(file_path, max_len=200, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Add delta and delta-delta features
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine features
    mfcc = np.concatenate([mfcc, delta, delta2])

    # Pad or crop
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

def build_dataset(base_path, extractor_fn, label_map, max_len):
    features = []
    labels = []

    for label_name, label in label_map.items():
        label_path = os.path.join(base_path, label_name)

        for person in os.listdir(label_path):
            person_path = os.path.join(label_path, person)

            for file in os.listdir(person_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(person_path, file)
                    try:
                        feature = extractor_fn(file_path, max_len=max_len)
                        features.append(feature)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes=3):
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First LSTM layer with bidirectional processing
        layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                       kernel_regularizer=regularizers.l2(0.001),
                                       recurrent_regularizer=regularizers.l2(0.001))),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second LSTM layer
        layers.Bidirectional(layers.LSTM(64,
                                       kernel_regularizer=regularizers.l2(0.001),
                                       recurrent_regularizer=regularizers.l2(0.001))),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Dense layers with residual connections
        layers.Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use Adam optimizer with a lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer=optimizer,
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Define paths and parameters
    base_path = "dataset"  # Update this to your dataset path
    max_len = 200
    n_mfcc = 13

    # Define label mapping
    label_map = {
        '13 Young Healthy Control': 0,
        '20 Elderly Healthy Control': 1,
        "22 People with Parkinson's disease": 2
    }

    # Build dataset
    print("Building dataset...")
    mfcc_features, mfcc_labels = build_dataset(base_path, extract_mfcc, label_map, max_len)
    
    # Visualize an example
    visualize_feature(mfcc_features[0], title="MFCC Example")

    # Prepare data for model - LSTM expects (samples, time_steps, features)
    X_mfcc = mfcc_features.transpose(0, 2, 1)  # Reshape for LSTM input
    
    # Normalize features
    scaler = StandardScaler()
    X_mfcc = scaler.fit_transform(X_mfcc.reshape(-1, X_mfcc.shape[-1])).reshape(X_mfcc.shape)

    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(X_mfcc, mfcc_labels, 
                                                       test_size=0.3, 
                                                       stratify=mfcc_labels, 
                                                       random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                                   test_size=0.5, 
                                                   stratify=y_temp, 
                                                   random_state=42)

    # Create and train model
    print("Creating and training model...")
    model = create_model(X_train.shape[1:])
    
    # Enhanced callbacks
    early_stop = EarlyStopping(monitor='val_loss', 
                              patience=10, 
                              restore_best_weights=True,
                              verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.2,
                                 patience=5,
                                 min_lr=0.00001,
                                 verbose=1)
    
    history = model.fit(X_train, y_train,
                       epochs=100,  # Increased epochs with early stopping
                       batch_size=32,
                       validation_data=(X_val, y_val),
                       callbacks=[early_stop, reduce_lr])

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_labels))

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_test, y_pred_labels), 
                annot=True, 
                fmt='d',
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the model
    print("\nSaving model...")
    model.save('models/lstm_model.h5')
    print("Model saved successfully!")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 