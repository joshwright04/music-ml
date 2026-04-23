import os
import librosa
import numpy as np
import soundfile as sf


class MelPreprocessor:
    def __init__(self, config):
        self.config = config

    def load_audio(self, file_path):
        try:
            y, _ = librosa.load(
                file_path,
                sr=self.config.sr,
                mono=True,
                duration=self.config.duration,
            )
        except Exception as e:
            print(f"Skipping unreadable file: {file_path}")
            print(f"Reason: {e}")
            return None

        desired_len = int(self.config.sr * self.config.duration)

        if len(y) < desired_len:
            y = np.pad(y, (0, desired_len - len(y)))
        else:
            y = y[:desired_len]

        return y

    def audio_to_mel(self, y):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            power=2.0,
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max,
            top_db=self.config.top_db,
        )

        mel_norm = (mel_db + self.config.top_db) / self.config.top_db
        return mel_norm.astype(np.float32)

    def preprocess_file(self, file_path):
        y = self.load_audio(file_path)

        if y is None:
            return None

        mel = self.audio_to_mel(y)
        return np.expand_dims(mel, axis=-1)

    def mel_to_audio(self, mel_norm, out_path):
        mel_norm = np.clip(mel_norm, 0.0, 1.0)

        mel_db = mel_norm * self.config.top_db - self.config.top_db
        mel_power = librosa.db_to_power(mel_db, ref=1.0)

        y = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=self.config.sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.n_fft,
            power=2.0,
            n_iter=64,
            length=int(self.config.sr * self.config.duration),
        )

        sf.write(out_path, y, self.config.sr)
        print(f"Saved {out_path}")


def find_audio_files(audio_dir):
    audio_files = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))

    return audio_files


def build_dataset(audio_files, preprocessor):
    processed = []
    skipped = 0

    for file_path in audio_files:
        x = preprocessor.preprocess_file(file_path)

        if x is None:
            skipped += 1
            continue

        processed.append(x)

    X = np.array(processed, dtype=np.float32)
    return X, skipped