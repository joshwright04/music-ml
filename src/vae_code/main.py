import os
# Suppress tensorflow logging, which can get messy and annoying
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from config import VAEConfig
from preprocessing import (
    MelPreprocessor,
    find_audio_files,
    build_dataset,
)
from vae import Encoder, Decoder
from train_vae import train_vae
from generate import (
    reconstruct_example,
    generate_example,
)

def get_desired_training_genre(config: VAEConfig) -> str:
    while True:
        genre = input(
            "Enter genre to train on "
            "(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock): "
        ).strip().lower()

        valid_genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]

        if genre in valid_genres:
            print(f"Audio filepath set to: ./data/Data/genres_original/{genre}")
            return f"./data/Data/genres_original/{genre}"

        print("Invalid genre. Please choose a valid genre.")

def get_desired_training_epochs(config: VAEConfig) -> str:
     while True:
            try:
                input_epochs = int(input("Enter number of epochs: "))

                if input_epochs <= 0:
                    print("Epochs must be greater than 0.")
                    continue

                print("Training Epochs set to ", input_epochs)
                return input_epochs

            except ValueError:
                print("Please enter a valid integer.")

def main():
    # Load the default config variables
    config = VAEConfig()

    config.audio_dir = get_desired_training_genre(config)

    config.epochs = get_desired_training_epochs(config)

    preprocessor = MelPreprocessor(config)

    audio_files = find_audio_files(config.audio_dir)
    print("Found", len(audio_files), "audio files from the path ", config.audio_dir)

    X, skipped = build_dataset(
        audio_files,
        preprocessor,
    )

    print("Usable files:", len(X))
    print("Skipped files:", skipped)

    if len(X) == 0:
        raise ValueError("No usable files found.")

    print("Input shape:", X.shape[1:])

    h, w = X.shape[1], X.shape[2]

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(X)
        .shuffle(len(X))
        .batch(config.batch_size)
    )

    encoder = Encoder(config.latent_dim)
    decoder = Decoder(
        config.latent_dim,
        output_shape=(h, w),
    )

    train_vae(
        encoder,
        decoder,
        dataset,
        config,
    )

    reconstruct_example(
        encoder,
        decoder,
        X[0],
        preprocessor,
    )

    generate_example(
        decoder,
        config,
        preprocessor,
    )


if __name__ == "__main__":
    main()