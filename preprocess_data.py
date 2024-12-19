import librosa
import numpy as np
import pandas as pd

# will use a "seconds" parameter to figure out how many chunks to use 
# similar to features_3sec from original gtzan, but with more options
# so for example if seconds is 5, we will split the audio file into 30/5=6 data points 
# default is 30 seconds where the entire audio file is a single data point
def extract_features(audio_path, file_name, seconds=30):
    num_samples = int(30 // seconds)

    all_features = []

    for i in range(num_samples):
        offset = seconds * i 
        duration = seconds 
        # make sure to include the very sall portion at the end which would've been lost since duration could lead to an upper bound
        # might not make a difference but just in case
        # librosa handles exceeding file length gracefully by just stopping 
        if offset + duration >= 30:
            duration += 0.00001

        # load file and do fourier transform
        y, sr = librosa.load(audio_path, offset = offset, duration = duration)
        D = librosa.stft(y)
        
        # use things like mean, var to get general information about the data and keep it smaller as well

        # trying to reproduce some gtzan features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        try:
            tempo = tempo[0]
        except TypeError:
            pass
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        chroma_stft_mean = np.mean(chroma)
        chroma_stft_var = np.var(chroma)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_var = np.var(spectral_centroid)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_var = np.var(spectral_bandwidth)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_var = np.mean(spectral_rolloff)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_var = np.var(zero_crossing_rate)
        harmony_mean = np.mean(y_harmonic)
        harmony_var = np.var(y_harmonic)
        perceptr_mean = np.mean(y_percussive)
        perceptr_var = np.var(y_percussive)

        # features which are not in gtzan
        # we have tonnetz, magnitude, spectral contrast, spectral flatness
        magnitude = librosa.amplitude_to_db(abs(D), ref=np.max)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(S=np.abs(D), sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        
        magnitude_mean = np.mean(magnitude)
        magnitude_var = np.var(magnitude)
        tonnetz_mean = np.mean(tonnetz)
        tonnetz_var = np.var(tonnetz)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
        spectral_contrast_var = np.var(spectral_contrast, axis=1)
        spectral_flatness_mean = np.mean(spectral_flatness)
        spectral_flatness_var = np.var(spectral_flatness)

        # features not in gtzan and not in librosa standard set
        # extracting features from short-time Fourier transform
        row_norms = np.linalg.norm(D, axis = 1)
        sorted_norms = np.argsort(row_norms)[::-1]
        highest_10_frequency_avg = np.mean(sorted_norms[:10])
        highest_20_frequency_avg = np.mean(sorted_norms[:20])
        highest_50_frequency_avg = np.mean(sorted_norms[:50])
        highest_100_frequency_avg = np.mean(sorted_norms[:100])
        highest_200_frequency_avg = np.mean(sorted_norms[:200])

        highest_10_frequency_var = np.var(sorted_norms[:10])
        highest_20_frequency_var = np.var(sorted_norms[:20])
        highest_50_frequency_var = np.var(sorted_norms[:50])
        highest_100_frequency_var = np.var(sorted_norms[:100])
        highest_200_frequency_var = np.var(sorted_norms[:200])
        
        feature = {
            "file_name": file_name,
            "magnitude_mean": magnitude_mean,
            "magnitude_var": magnitude_var,
            "chroma_stft_mean": chroma_stft_mean,
            "chroma_stft_var": chroma_stft_var,
            "rms_mean": rms_mean,
            "rms_var": rms_var,
            "spectral_centroid_mean": spectral_centroid_mean,
            "spectral_centroid_var": spectral_centroid_var,
            "spectral_bandwidth_mean": spectral_bandwidth_mean,
            "spectral_bandwidth_var": spectral_bandwidth_var,
            "rolloff_mean": spectral_rolloff_mean,
            "rolloff_var": spectral_rolloff_var,
            "zero_crossing_rate_mean": zero_crossing_rate_mean,
            "zero_crossing_rate_var": zero_crossing_rate_var,
            "harmony_mean": harmony_mean,
            "harmony_var": harmony_var,
            "perceptr_mean": perceptr_mean,
            "perceptr_var": perceptr_var,
            "tempo": tempo,
            "tonnetz_mean": tonnetz_mean,
            "tonnetz_var": tonnetz_var,
            "spectral_flatness_mean": spectral_flatness_mean,
            "spectral_flatness_std": spectral_flatness_var,
            **{f"mfcc{i+1}_mean": np.mean(mfccs[i]) for i in range(mfccs.shape[0])},
            **{f"mfcc{i+1}_var": np.var(mfccs[i]) for i in range(mfccs.shape[0])},
            **{f"spectral_contrast_{i+1}_mean": spectral_contrast_mean[i] for i in range(len(spectral_contrast_mean))},
            **{f"spectral_contrast_{i+1}_std": spectral_contrast_var[i] for i in range(len(spectral_contrast_var))}
        }

        all_features.append(feature)
    return all_features
    

audio_path = 'datasets/gtzan/genres_original/blues/blues.00000.wav'  
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

base_path = 'datasets/gtzan/genres_original'
features_list = []
count = 0

# the amount of seconds each sample will be 
# will be doing 30 seconds(full file) along with 15, 10, 5, 3, 2, 1
splits_seconds = [2, 1]

# we skip jazz 54 since that is known to be corrupted
for s in splits_seconds:
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
            feature = extract_features(full_path, file_name, s)
            for f in feature:
                f['genre'] = g 
                features_list.append(f)

    print(len(features_list))
    df = pd.DataFrame(features_list)
    df.to_csv(f'datasets/gtzan_processed_{s}s/processed_features.csv', index=False)
    # reset features list for next set of samples
    # if doing all combined then comment out the reset and move the saving to csv lines above this outside the loop
    features_list = []
