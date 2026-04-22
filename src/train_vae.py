import os

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import Model, layers


SR = 22050
DURATION = 3
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
TOP_DB = 80.0
LATENT_DIM = 32
BATCH_SIZE = 16
EPOCHS = 30


def audio_to_mel(file, sr=SR, duration=DURATION):
    try:
        y, _ = librosa.load(file, sr=sr, mono=True, duration=duration)
    except Exception as e:
        print(f"Skipping unreadable file: {file}")
        print(f"Reason: {e}")
        return None

    desired_len = int(sr * duration)
    if len(y) < desired_len:
        y = np.pad(y, (0, desired_len - len(y)))
    else:
        y = y[:desired_len]

    # librosa docs: melspectrogram computes a mel-scaled spectrogram
    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    M_db = librosa.power_to_db(M, ref=np.max, top_db=TOP_DB)

    # map [-TOP_DB, 0] -> [0, 1]
    M_norm = (M_db + TOP_DB) / TOP_DB
    return M_norm.astype(np.float32)


def mel_shape(sr=SR, duration=DURATION):
    dummy = np.zeros(int(sr * duration), dtype=np.float32)
    M = librosa.feature.melspectrogram(
        y=dummy,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    return M.shape  # (n_mels, time_frames)


def mel_to_audio(M_norm, out_path, sr=SR):
    M_norm = np.clip(M_norm, 0.0, 1.0)

    # map [0,1] -> [-TOP_DB, 0]
    M_db = M_norm * TOP_DB - TOP_DB

    # invert dB -> power
    M_power = librosa.db_to_power(M_db, ref=1.0)

    # librosa docs: mel_to_audio inverts a mel power spectrogram to audio using Griffin-Lim
    y = librosa.feature.inverse.mel_to_audio(
        M_power,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2.0,
        n_iter=64,
        length=int(sr * DURATION),
    )

    sf.write(out_path, y, sr)
    print(f"Saved {out_path}")


def preprocess_file(file):
    M = audio_to_mel(file)
    if M is None:
        return None
    return np.expand_dims(M, axis=-1)  # (n_mels, time_frames, 1)


class Encoder(Model):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")
        self.conv3 = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")
        self.flatten = layers.Flatten()
        self.mu = layers.Dense(latent_dim)
        self.logvar = layers.Dense(latent_dim)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.mu(x), self.logvar(x)


def sample_z(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


class Decoder(Model):
    def __init__(self, latent_dim=LATENT_DIM, output_shape=None):
        super().__init__()
        self.output_shape_hw = output_shape  # (H, W)

        h, w = output_shape
        self.start_h = int(np.ceil(h / 8))
        self.start_w = int(np.ceil(w / 8))

        self.fc = layers.Dense(self.start_h * self.start_w * 128, activation="relu")
        self.reshape_layer = layers.Reshape((self.start_h, self.start_w, 128))
        self.deconv1 = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")
        self.deconv2 = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")
        self.deconv3 = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")
        self.out = layers.Conv2D(1, 3, padding="same", activation="sigmoid")

    def call(self, z, training=False):
        x = self.fc(z)
        x = self.reshape_layer(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.out(x)

        # crop to exact target shape
        h, w = self.output_shape_hw
        x = x[:, :h, :w, :]
        return x


def vae_loss(x, x_recon, mu, logvar, beta=1e-4):
    recon = tf.reduce_mean(tf.square(x - x_recon))
    kl = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return recon + beta * kl, recon, kl


def reconstruct_example(encoder, decoder, x, out_path="reconstruction.wav"):
    x_tensor = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    mu, logvar = encoder(x_tensor, training=False)
    z = mu
    M_recon = decoder(z, training=False).numpy()[0, :, :, 0]
    mel_to_audio(M_recon, out_path)


def generate_example(decoder, out_path="generated.wav"):
    z = tf.random.normal((1, LATENT_DIM))
    M_gen = decoder(z, training=False).numpy()[0, :, :, 0]
    mel_to_audio(M_gen, out_path)


def main():
    AUDIO_DIR = "./data/Data/genres_original"

    audio_files = []
    for root, _, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    print("Found", len(audio_files), "audio files")

    processed = []
    skipped = 0

    for f in audio_files:
        x = preprocess_file(f)
        if x is None:
            skipped += 1
            continue
        processed.append(x)

    X = np.array(processed, dtype=np.float32)
    print("Usable files:", len(X))
    print("Skipped files:", skipped)

    h, w = X.shape[1], X.shape[2]
    print("Input shape:", X.shape[1:])

    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X)).batch(BATCH_SIZE)

    encoder = Encoder(latent_dim=LATENT_DIM)
    decoder = Decoder(latent_dim=LATENT_DIM, output_shape=(h, w))
    optimizer = tf.keras.optimizers.Adam(1e-3)

    for epoch in range(EPOCHS):
        losses = []
        recons = []
        kls = []

        for batch in dataset:
            with tf.GradientTape() as tape:
                mu, logvar = encoder(batch, training=True)
                z = sample_z(mu, logvar)
                x_recon = decoder(z, training=True)
                loss, recon, kl = vae_loss(batch, x_recon, mu, logvar)

            vars_ = encoder.trainable_variables + decoder.trainable_variables
            grads = tape.gradient(loss, vars_)
            optimizer.apply_gradients(zip(grads, vars_))

            losses.append(float(loss.numpy()))
            recons.append(float(recon.numpy()))
            kls.append(float(kl.numpy()))

        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"recon={np.mean(recons):.4f} | "
            f"kl={np.mean(kls):.4f}"
        )

    reconstruct_example(encoder, decoder, X[0], out_path="reconstruction.wav")
    generate_example(decoder, out_path="generated.wav")


if __name__ == "__main__":
    main()