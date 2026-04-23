import tensorflow as tf


def reconstruct_example(
    encoder,
    decoder,
    sample,
    preprocessor,
    out_path="reconstruction.wav",
):
    x = tf.convert_to_tensor(
        sample[None, ...],
        dtype=tf.float32,
    )

    mu, _ = encoder(x)
    reconstructed = decoder(mu).numpy()[0, :, :, 0]

    preprocessor.mel_to_audio(
        reconstructed,
        out_path,
    )


def generate_example(
    decoder,
    config,
    preprocessor,
    out_path="generated.wav",
):
    z = tf.random.normal((1, config.latent_dim))

    generated = decoder(z).numpy()[0, :, :, 0]

    preprocessor.mel_to_audio(
        generated,
        out_path,
    )