# CSCI 4622 - Music Genre Classification and Generation

Gavin Petruzzi and Joshua Wright

## Project Overview

This project explores music genre classification using both traditional machine learning models and a Variational Autoencoder (VAE) for music generation.

The project has two main goals:

1. Train baseline classification models to identify the genre of songs from extracted audio features.
2. Train a VAE on genre-specific audio samples to generate new songs, then evaluate whether those generated songs preserve genre characteristics by passing them through the baseline classifiers.

This creates both a generation pipeline and a quantitative evaluation method for generated audio.

---

## Models Used

### Baseline Classification Models

The following supervised learning models are trained using the GTZAN feature dataset (`features_30_sec.csv`):

* Random Forest
* Logistic Regression
* Support Vector Machine (SVM)

These models are trained on extracted numerical audio features such as:

* MFCCs
* Spectral Centroid
* Spectral Bandwidth
* Rolloff
* Zero Crossing Rate
* Chroma Features
* RMS Energy
* Tempo
* Harmony / Percussive Features

These models are saved using `joblib` and later reused to evaluate VAE-generated songs.

---

### Variational Autoencoder (VAE)

The VAE is trained on mel spectrograms generated from `.wav` audio files.

Pipeline:

```text
Audio (.wav)
→ Mel Spectrogram
→ Encoder
→ Latent Space
→ Decoder
→ Reconstructed / Generated Audio
```

Users can:

* Train a new genre-specific VAE
* Load a previously saved VAE model
* Generate multiple songs from the trained latent space
* Save model weights for future reuse

Generated songs are converted back into `.wav` format using Griffin-Lim inversion.

---

## Evaluation of Generated Songs

Generated songs are evaluated by:

```text
Generated WAV files
→ Feature Extraction
→ generated_features.csv
→ Baseline Classifiers
→ Predicted Genre
```

If the baseline models classify generated songs as the intended genre (for example, a jazz-trained VAE producing songs classified as jazz), this provides quantitative evidence that the VAE preserved genre-specific structure.

A PNG bar chart is also generated showing baseline model accuracy on generated songs.

---

## Setup

Recommended Python Version:

Python 3.10+

A virtual environment is strongly recommended for this project before installing dependencies.

### Create and Activate a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Dependencies

Once the virtual environment is activated, install dependencies with:

```bash
pip install -r requirements.txt
```

### Dataset

This project uses the GTZAN Genre Collection dataset. It is far too large to include in this repo.

Expected files include:

```text
./data/Data/features_30_sec.csv
./data/Data/genres_original/*
```

The dataset must be present for the project to run correctly.
It can be found at:
[https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

Place the zip file from the above link into the `data` folder and unzip it. The final structure should look like this:

```
data/
└── Data/
    ├── genres_original/
    ├── images_original/
    ├── features_30_sec.csv
    └── features_3_sec.csv
```

---

## Project Structure

```text
music-ml/
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── baseline_models/
│   │   ├── run_baseline_models.py
│   │   └── evaluate_baseline_models.py
│   │
│   └── vae_code/
│       ├── main.py
│       ├── preprocessing.py
│       ├── train_vae.py
│       ├── generate.py
│       └── utilities/
│
├── data/ (*data directory initially empty - See description below for a link to download the GTZAN dataset*)
│   └── Data/
|        ...
│
├── saved_baseline_models/ (*directory initially empty*)
├── saved_vae_models/ (*directory initially empty*)
└── generated_songs/ (*directory initially empty*)
```

---

## How to Run

### 1. Activate Your venv

Activate your venv if not done already:

For Windows: 

```bash
venv\Scripts\activate
```

For Mac/Linux: 

```bash
source venv/bin/activate
```

### 2. Train Baseline Models

```bash
python src/baseline_models/run_baseline_models.py
```

This will:

* Train Random Forest, Logistic Regression, and SVM
* Automatically save these three trained models into `saved_baseline_models/`
* Generate model comparison plots into `original_baseline_plots/`

---

### 3. Run the VAE Pipeline

```bash
python src/vae_code/main.py
```

This will allow the user to:

* Train a new VAE model OR load an existing saved model
    * The repo contains no saved models originally. Saved vae models can be found as .h5 files in the directory `saved_vae_models/`
* Select genre and training epochs
* Generate new 3 second song snippets  
* Extract generated features
* Evaluate generated songs using baseline models
* Save generated graphs and optionally save model weights

---
