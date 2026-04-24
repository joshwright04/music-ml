import tensorflow as tf
 

def generate_example(
    decoder,
    config,
    preprocessor,
    out_path="generated.wav",
) -> None:
    """
    Generate a new song from the vae decoder
    and save it as a .wav file
    """

    # Sampling a random vector from latent space
    z = tf.random.normal((1, config.latent_dim))

    # Decode latent vector into a mel spectrogram
    # Mel frequency is important to use here becuase it represents audio in a way that mimics human hearing perception
    # Deep dive on this info can be found here: 
    # https://medium.com/@MuhyEddin/feature-extraction-is-one-of-the-most-important-steps-in-developing-any-machine-learning-or-deep-94cf33a5dd46
    generated = decoder(z).numpy()[0, :, :, 0]

    # Convert mel spectrogram back into audio as a .wav file
    preprocessor.mel_to_audio(
        generated,
        out_path,
    )