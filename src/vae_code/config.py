from dataclasses import dataclass

# Dataclass VAEConfig as default config values for our vae.
# Some of these values are changed based on user input in vae_code/main.py.
# Specifically: epochs and audio_dir 
@dataclass
class VAEConfig:
    sr: int = 22050
    duration: int = 3

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 128
    top_db: float = 80.0

    latent_dim: int = 32

    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 1e-3

    audio_dir: str = "./data/Data/genres_original"