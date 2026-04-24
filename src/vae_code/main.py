import os
# Suppress tensorflow logging in the terminal, which can get messy and annoying
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from config import VAEConfig
from utilities.extract_generated_features import generate_csv
from utilities.load_baseline_models import load_baseline_models
from utilities.evaluate_vae_output import evaluate_vae_output_on_baseline_models
from utilities.clear_output_directory import clear_output_directory
from preprocessing import (
    MelPreprocessor,
    find_audio_files,
    build_dataset,
)
from vae import Encoder, Decoder
from train_vae import train_vae
from generate import (
    # Reconstruction removed but might be cool to mess with in the future
    generate_example
)

def get_desired_training_genre(config: VAEConfig) -> str:
    """
    Get user input for genre to train on. Loops until a valid genre is selected. 
    Returns the updated config.audio_dir
    """
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
    """
    Get user input for # of epochs to train on. Loops until a valid number is selected. 
    Returns the updated config.epochs
    """
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
    """
    Get user input for # of 3 second song snippets to generate. Loops until a valid number is selected. 
    Returns the number of songs to generate as an int.
    """
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
    """
    Get user input if trained VAEs should be saved or not.
    Saves trained models as .h5 files in the 'saved_vae_models' directory
    """
    while True:
        save_model = input("Do you want to save the trained models? (y/n): ").strip().lower()

        if save_model == "y":
            # Create directory of one does not alrady exist
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
    
def load_existing_vae_models() -> tuple[str, str, str]:
    """
    Gets user input to load a saved VAE model. Will loop until a valid selection is made

    Returns the path to the encoder, the path to the associated decoder, and the genre 
    so that config.audio_dir can be updated in the main function
    """
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

    # Create a Mel Preprocessor
    preprocessor = MelPreprocessor(config)

    # Since we expect all files in our data to be of the same shape, we can sample the first 
    # .wav file to get placeholder values for h and w 
    dummy_sample = preprocessor.preprocess_file(
        "./data/Data/genres_original/blues/blues.00000.wav"
    )

    # Set the aforementioned h and w 
    h, w = dummy_sample.shape[0], dummy_sample.shape[1]

    # Loop for user selection whether to train a new model or load an existing one
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
        # User wants to train a new model, call user input functions defined above
        config.audio_dir = get_desired_training_genre(config)

        config.epochs = get_desired_training_epochs()

        # Rebuild preprocessor for config changes
        preprocessor = MelPreprocessor(config)

        # Load the audio files in the audio_dir
        audio_files = find_audio_files(config.audio_dir)
        print("Found", len(audio_files), "audio files from the path ", config.audio_dir)

        # Build the dataset
        X, skipped = build_dataset(
            audio_files,
            preprocessor,
        )

        print("Usable files:", len(X))
        print("Skipped files:", skipped)

        if len(X) == 0:
            raise ValueError("No usable files found.")

        print("Input shape:", X.shape[1:])

        # Create the dataset used for training, shuffle and batch for randomization and improved training speed
        dataset = (
            tf.data.Dataset
            .from_tensor_slices(X)
            .shuffle(len(X))
            .batch(config.batch_size)
        )

        # Initialize the VAE encoder and decoder
        encoder = Encoder(config.latent_dim)
        decoder = Decoder(
            config.latent_dim,
            output_shape=(h, w),
        )

        # Train the VAE
        train_vae(
            encoder,
            decoder,
            dataset,
            config,
        )

    else:
        # Load existing model 
        genre, encoder_path, decoder_path = load_existing_vae_models()
        config.audio_dir = f"./data/Data/genres_original/{genre}"

         # Rebuild preprocessor
        preprocessor = MelPreprocessor(config)

        # Initialize new encoders and decoders, weights will be updated with the .h5 files after
        encoder = Encoder(config.latent_dim)
        decoder = Decoder(
            config.latent_dim,
            output_shape=(h, w)
        )

        # Needed to set to summy values before updating to loaded weights
        dummy_x = tf.zeros((1, h, w, 1))
        encoder(dummy_x)

        dummy_z = tf.zeros((1, config.latent_dim))
        decoder(dummy_z)

        # Load model weights from .h5 files
        encoder.load_weights(encoder_path)
        decoder.load_weights(decoder_path)

        print("Model loaded successfully.")

    # Now that we have our trained VAE (in either case), get user input on number of songs to generate
    num_songs_to_generate = get_desired_songs_to_generate()

    # Get genre name and generated output directory path
    genre_name = os.path.basename(config.audio_dir)
    output_dir = f"generated_songs/{genre_name}_songs"

    # Create this directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # Clear the output directory in case leftover .wav files, csv files, or plots still exist in it
    clear_output_directory(output_dir)
 
    # Generate 'num_songs_to_generate' songs with the VAE
    for i in range(num_songs_to_generate):
        generate_example(
            decoder,
            config,
            preprocessor,
            f"{output_dir}/generated_{genre_name}_{i + 1}.wav"
        )

    # Generate a csv with each row being a generated .wav file. We want the mel spectrogram features
    # extracted here using librosa so we can predict the genres with our original baseline models
    print("Creating a features csv file for the generated songs...")
    generate_csv(output_dir)

    # Load the baseline models from 'saved_baseline_models' after baseline_models/run_baseline_models.py script has been run
    baseline_models = load_baseline_models()

    # Evaluate each generated song on each of the three baseline models
    evaluate_vae_output_on_baseline_models(baseline_models, 
                                           os.path.join(output_dir, "generated_features.csv"), 
                                           genre_name, 
                                           output_dir
    )

    # Only ask to save model if it is not already saved
    if train_new_model:
        save_models(config, encoder, decoder)

    # Script done

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Critical Error Occured: {e}")