import librosa
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
import cv2 
from tensorflow.keras import layers, Model
import os

def preprocess_file(file):
    S = audio_to_spectrogram(file)
    S_resized = cv2.resize(S, (128,128))
    return np.expand_dims(S_resized, -1)  # add channel dimension

def audio_to_spectrogram(file, n_fft=2048, hop_length=512, duration=3):
    # load with pydub
    audio = AudioSegment.from_file(file)
    audio = audio.set_channels(1).set_frame_rate(22050)  # mono, 22050 Hz
    # get raw samples as numpy float32
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / (1 << 15)
    # truncate/pad to duration
    desired_len = int(22050 * duration)
    if len(y) > desired_len:
        y = y[:desired_len]
    else:
        y = np.pad(y, (0, desired_len - len(y)))
    # compute spectrogram
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
    return S_norm.astype(np.float32)

latent_dim = 16  # small for quick testing
input_shape = (128, 128, 1)  # spectrograms resized

# Encoder
class Encoder(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')
        self.conv2 = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.fc_mu = layers.Dense(latent_dim)
        self.fc_logvar = layers.Dense(latent_dim)
        
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Reparameterization trick
def sample_z(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps

# Decoder
class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.fc = layers.Dense(32*32*64, activation='relu')
        self.reshape_layer = layers.Reshape((32,32,64))
        self.deconv1 = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')
        self.deconv2 = layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
    
    def call(self, z):
        x = self.fc(z)
        x = self.reshape_layer(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x

# Loss function
def vae_loss(x, x_recon, mu, logvar):
    recon_loss = tf.reduce_mean(tf.keras.losses.mse(x, x_recon))
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mu) - tf.exp(logvar))
    return recon_loss + kl_loss

def main():
    
    AUDIO_DIR = "./data/Data/genres_original"

    # get a list of all .wav files
    audio_files = []
    for root, dirs, files in os.walk(AUDIO_DIR):
        for f in files:
            if f.endswith(".wav"):
                audio_files.append(os.path.join(root, f))

    print("Found", len(audio_files), "audio files")

    X = np.array([preprocess_file(f) for f in audio_files])
    dataset = tf.data.Dataset.from_tensor_slices(X).shuffle(100).batch(16)

    encoder = Encoder()
    decoder = Decoder()
    optimizer = tf.keras.optimizers.Adam(1e-3)

    for epoch in range(10):
        for batch in dataset:
            with tf.GradientTape() as tape:
                mu, logvar = encoder(batch)
                z = sample_z(mu, logvar)
                x_recon = decoder(z)
                loss = vae_loss(batch, x_recon, mu, logvar)
            grads = tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
        print(f"Epoch {epoch+1}, loss: {loss.numpy():.4f}")

if __name__ == "__main__":
    main()