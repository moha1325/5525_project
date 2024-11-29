import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

# one audio file 
audio_path = 'datasets/gtzan/genres_original/blues/blues.00000.wav'  
y, sr = librosa.load(audio_path, sr=None)  

features = []

# do short-time Fourier transform
D = librosa.stft(y)

# convert to magnitutde
magnitude = librosa.amplitude_to_db(abs(D), ref=np.max)

# also mfcc and chroma features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)

features.append((magnitude.flatten(), mfccs, chroma))

plt.figure(figsize=(10, 6))
librosa.display.specshow(magnitude, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.savefig('spectrogram.png')

print(features)

end_time = time.time()

# for development purposes
total_time = end_time - start_time

print(f"{total_time} seconds taken to process an audio file")