from __future__ import annotations

from pathlib import Path
import numpy as np
import librosa


def extract_features(file_path: str | Path) -> dict[str, float]:
    """
    Extract a simple fixed-size feature vector from one audio file.
    Returns summary statistics so every song becomes one row of numbers.
    """
    file_path = Path(file_path)

    # Load ~30 seconds max, mono audio
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30.0)

    # Basic spectral / rhythmic features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)

    # MFCCs and chroma are very common for music/audio tasks
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    features: dict[str, float] = {
        "tempo": float(np.squeeze(tempo)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_centroid_std": float(np.std(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_bandwidth_std": float(np.std(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_rolloff_std": float(np.std(spectral_rolloff)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }

    for i in range(mfccs.shape[0]):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))

    for i in range(chroma.shape[0]):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    return features


if __name__ == "__main__":
    # quick sanity test
    sample_file = Path("data/example.wav")
    feats = extract_features(sample_file)
    for k, v in list(feats.items())[:10]:
        print(k, v)