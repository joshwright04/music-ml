import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class Encoder(Model):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = layers.Conv2D(
            32, 3, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv2D(
            64, 3, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv2D(
            128, 3, strides=2, padding="same", activation="relu"
        )

        self.flatten = layers.Flatten()
        self.mu = layers.Dense(latent_dim)
        self.logvar = layers.Dense(latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        return self.mu(x), self.logvar(x)


def sample_z(mu, logvar):
    eps = tf.random.normal(shape=tf.shape(mu))
    return mu + tf.exp(0.5 * logvar) * eps


class Decoder(Model):
    def __init__(self, latent_dim, output_shape):
        super().__init__()

        self.output_shape_hw = output_shape
        h, w = output_shape

        self.start_h = int(np.ceil(h / 8))
        self.start_w = int(np.ceil(w / 8))

        self.fc = layers.Dense(
            self.start_h * self.start_w * 128,
            activation="relu",
        )

        self.reshape_layer = layers.Reshape(
            (self.start_h, self.start_w, 128)
        )

        self.deconv1 = layers.Conv2DTranspose(
            128, 3, strides=2, padding="same", activation="relu"
        )
        self.deconv2 = layers.Conv2DTranspose(
            64, 3, strides=2, padding="same", activation="relu"
        )
        self.deconv3 = layers.Conv2DTranspose(
            32, 3, strides=2, padding="same", activation="relu"
        )

        self.out = layers.Conv2D(
            1, 3, padding="same", activation="sigmoid"
        )

    def call(self, z):
        x = self.fc(z)
        x = self.reshape_layer(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        x = self.out(x)

        h, w = self.output_shape_hw
        x = x[:, :h, :w, :]

        return x


def vae_loss(x, x_recon, mu, logvar, beta=1e-4):
    recon = tf.reduce_mean(tf.square(x - x_recon))

    kl = -0.5 * tf.reduce_mean(
        1 + logvar - tf.square(mu) - tf.exp(logvar)
    )

    total = recon + beta * kl
    return total, recon, kl