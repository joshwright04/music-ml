import numpy as np
import tensorflow as tf
from vae import sample_z, vae_loss


def train_vae(encoder, decoder, dataset, config):
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    for epoch in range(config.epochs):
        losses = []
        recons = []
        kls = []

        for batch in dataset:
            with tf.GradientTape() as tape:
                mu, logvar = encoder(batch)
                z = sample_z(mu, logvar)

                x_recon = decoder(z)

                loss, recon, kl = vae_loss(
                    batch,
                    x_recon,
                    mu,
                    logvar,
                )

            variables = (
                encoder.trainable_variables
                + decoder.trainable_variables
            )

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            losses.append(float(loss.numpy()))
            recons.append(float(recon.numpy()))
            kls.append(float(kl.numpy()))

        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"recon={np.mean(recons):.4f} | "
            f"kl={np.mean(kls):.4f}"
        )