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
            print(f"Audio filepath set to: {config.audio_dir}/{genre}")
            return f"{config.audio_dir}/{genre}"

        print("Invalid genre. Please choose a valid genre.")

def get_desired_training_epochs() -> int:
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

def get_desired_songs_to_generate() -> int:
    while True:
        try:
            num_songs_to_generate = int(input("Enter number of songs to generate: "))

            if num_songs_to_generate <= 0:
                print("Number of songs must be greater than 0.")
                continue

            return num_songs_to_generate

        except ValueError:
            print("Please enter a valid integer.")

def save_models(config: VAEConfig, encoder: Encoder, decoder: Decoder) -> None:
    while True:
        save_model = input("Do you want to save the trained models? (y/n): ").strip().lower()

        if save_model in ["y", "yes"]:
            os.makedirs("saved_vae_models", exist_ok=True)

            genre_name = os.path.basename(config.audio_dir)

            encoder.save_weights(
                f"saved_vae_models/{genre_name}_encoder.weights.h5"
            )

            decoder.save_weights(
                f"saved_vae_models/{genre_name}_decoder.weights.h5"
            )

            print("Model weights saved successfully.")
            break

        elif save_model in ["n", "no"]:
            print("Model weights were not saved.")
            break

        else:
            print("Please enter y or n.")
    
def load_existing_models() -> tuple[str, str, str]:
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

    while True:
        genre = input(
            "Which saved model would you like to load? "
        ).strip().lower()

        if genre not in valid_genres:
            print("Invalid genre.")
            continue

        encoder_path = f"saved_vae_models/{genre}_encoder.weights.h5"
        decoder_path = f"saved_vae_models/{genre}_decoder.weights.h5"

        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            print(f"Found saved {genre} model.")
            return genre, encoder_path, decoder_path

        print(f"No saved model found for {genre}.")


def main():
    # Load the default config variables
    config = VAEConfig()

    preprocessor = MelPreprocessor(config)

    dummy_sample = preprocessor.preprocess_file(
        "./data/Data/genres_original/blues/blues.00000.wav"
    )

    h, w = dummy_sample.shape[0], dummy_sample.shape[1]

    while True:
        model_choice = input(
            "Do you want to (1) train a new model or (2) load an existing model? Enter 1 or 2: "
        ).strip()

        if model_choice == "1":
            train_new_model = True
            break

        elif model_choice == "2":
            train_new_model = False
            break

        else:
            print("Please enter 1 or 2.")

    if train_new_model:
        config.audio_dir = get_desired_training_genre(config)

        config.epochs = get_desired_training_epochs()

        # Rebuild preprocessor for config changes
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

    else:
    # Load existing model 
        genre, encoder_path, decoder_path = load_existing_models()
        config.audio_dir = f"./data/Data/genres_original/{genre}"
        preprocessor = MelPreprocessor(config)

        encoder = Encoder(config.latent_dim)
        decoder = Decoder(
            config.latent_dim,
            output_shape=(h, w)
        )

        dummy_x = tf.zeros((1, h, w, 1))
        encoder(dummy_x)

        dummy_z = tf.zeros((1, config.latent_dim))
        decoder(dummy_z)

        encoder.load_weights(encoder_path)
        decoder.load_weights(decoder_path)

        print("Model loaded successfully.")

    
    num_songs_to_generate = get_desired_songs_to_generate()

    genre_name = os.path.basename(config.audio_dir)
    output_dir = f"generated_songs/{genre_name}_songs"

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_songs_to_generate):
        generate_example(
            decoder,
            config,
            preprocessor,
            f"{output_dir}/generated_{genre_name}_{i + 1}.wav"
        )

    # reconstruct_example(
    #     encoder,
    #     decoder,
    #     X[0],
    #     preprocessor,
    # )

    # TODO: Pass generated songs into basic models to evaluate vae output accuracy

    # Only ask to save model if it is not already saved
    if train_new_model:
        save_models(config, encoder, decoder)

if __name__ == "__main__":
    main()