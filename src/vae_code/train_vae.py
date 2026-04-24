import numpy as np
import tensorflow as tf
from vae import sample_z, vae_loss

def train_vae(encoder, decoder, dataset, config):
    """
    Train the VAE using batches of mel spectrograms.

    Each batch is encoded into a latent distribution,
    sampled as a latent vector, decoded back into a
    mel spectrogram, and the model's performance
    is evaluated with the loss function (as mentioned in 
    vae.py)

    This loss function is based on reconstruction error and KL divergence
    """
    # Adam optimizer for updating weights in the encoder/decoder
    optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    # Loop through each epoch
    for epoch in range(config.epochs):
        losses = []
        recons = []
        kls = []

        for batch in dataset:
            # For each batch in the dataset, encode as a latent distribution
            with tf.GradientTape() as tape:
                mu, logvar = encoder(batch)
                z = sample_z(mu, logvar)

                # Sample latent vector
                x_recon = decoder(z)

                # Compute loss
                loss, recon, kl = vae_loss(
                    batch,
                    x_recon,
                    mu,
                    logvar,
                )

            # Combine the trainable weights of the encoder and decoder models
            variables = (
                encoder.trainable_variables
                + decoder.trainable_variables
            )

            # Optimize these weights using the gradient of the loss
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            # Save these loss values to print
            losses.append(float(loss.numpy()))
            recons.append(float(recon.numpy()))
            kls.append(float(kl.numpy()))

        # For each epoch, print averages for:
        # Reconstruction loss
        # Kl divergence
        # Combined total loss
        print(
            f"Epoch {epoch + 1:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"recon={np.mean(recons):.4f} | "
            f"kl={np.mean(kls):.4f}"
        )