import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd 

start_time = time.time()


def extract_features(audio_path, file_name):
    y, sr = librosa.load(audio_path, sr=None)
    # apply the fourier transform
    D = librosa.stft(y)

    # magnitude, mfccs, and chroma features
    magnitude = librosa.amplitude_to_db(abs(D), ref=np.max)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    # use things like mean, std to keep amount of data lower
    magnitude_mean = np.mean(magnitude)
    magnitude_std = np.std(magnitude)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    return {
        "file_name": file_name,
        "magnitude_mean": magnitude_mean,
        "magnitude_std": magnitude_std,
        **{f"mfcc_{i+1}_mean": mfccs_mean[i] for i in range(len(mfccs_mean))},
        **{f"mfcc_{i+1}_std": mfccs_std[i] for i in range(len(mfccs_std))},
        **{f"chroma_{i+1}_mean": chroma_mean[i] for i in range(len(chroma_mean))},
        **{f"chroma_{i+1}_std": chroma_std[i] for i in range(len(chroma_std))}
    }


audio_path = 'datasets/gtzan/genres_original/blues/blues.00000.wav'  
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

base_path = 'datasets/gtzan/genres_original'
features_list = []
count = 0


# we skip jazz 54 since that is known to be corrupted

for g in genres:
    for i in range(100):
        num = f"000{str(i)}"
        if i < 10:
            num = f"0000{str(i)}"
        
        if i == 54 and g == "jazz":
            continue
        
        file_name = f"/{g}/{g}.{num}.wav"
        full_path = base_path + file_name
        print(full_path)
        feature = extract_features(full_path, file_name)
        feature['genre'] = g 
        features_list.append(feature)

print(len(features_list))
df = pd.DataFrame(features_list)
df.to_csv('datasets/gtzan_processed/processed_features.csv', index=False)

