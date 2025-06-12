import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_audio(file_path, max_len=200, n_mfcc=13):
    """
    Process audio file for prediction.
    Args:
        file_path: Path to the audio file
        max_len: Maximum length of MFCC features
        n_mfcc: Number of MFCC coefficients
    Returns:
        Processed audio features ready for LSTM model prediction
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=22050)
    
    # Extract MFCC features
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
    
    # Reshape for LSTM input (samples, time_steps, features)
    mfcc = mfcc.transpose(1, 0)  # Transpose to (time_steps, features)
    
    # Normalize features
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)
    
    # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=0)
    
    return mfcc 