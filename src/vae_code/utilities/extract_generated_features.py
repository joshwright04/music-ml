import os
import librosa
import numpy as np
import pandas as pd

def generate_csv(output_dir: str) -> None:
    rows = []

    for file_name in os.listdir(output_dir):
        if not file_name.endswith(".wav"):
            continue

        file_path = os.path.join(output_dir, file_name)
        rows.append(extract_features(file_path))

    if not rows:
        print("No .wav files found in the directory.")
        return

    df = pd.DataFrame(rows)

    output_csv_path = os.path.join(output_dir, "generated_features.csv")
    df.to_csv(output_csv_path, index=False)

    print(f"Saved {output_csv_path}")
    df.to_csv("generated_features.csv", index=False)
    print("Saved generated_features.csv")

def extract_features(file_path: str) -> dict:
    y, sr = librosa.load(file_path, sr=None)

    features = {}

    features["filename"] = os.path.basename(file_path)
    features["length"] = int(len(y))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features["chroma_stft_mean"] = float(np.mean(chroma))
    features["chroma_stft_var"] = float(np.var(chroma))

    features["rms_mean"] = float(np.mean(rms))
    features["rms_var"] = float(np.var(rms))

    features["spectral_centroid_mean"] = float(np.mean(spec_cent))
    features["spectral_centroid_var"] = float(np.var(spec_cent))

    features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
    features["spectral_bandwidth_var"] = float(np.var(spec_bw))

    features["rolloff_mean"] = float(np.mean(rolloff))
    features["rolloff_var"] = float(np.var(rolloff))

    features["zero_crossing_rate_mean"] = float(np.mean(zcr))
    features["zero_crossing_rate_var"] = float(np.var(zcr))

    features["harmony_mean"] = float(np.mean(harmony))
    features["harmony_var"] = float(np.var(harmony))

    features["perceptr_mean"] = float(np.mean(perceptr))
    features["perceptr_var"] = float(np.var(perceptr))

    features["tempo"] = float(np.squeeze(tempo))

    for i in range(20):
        features[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc{i+1}_var"] = float(np.var(mfcc[i]))

    features["label"] = "generated"

    return features