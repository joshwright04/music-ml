from __future__ import annotations

from pathlib import Path
import pandas as pd

from features import extract_features


DATA_DIR = Path("data/genres")
OUTPUT_CSV = Path("results/features.csv")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".au", ".flac", ".ogg", ".m4a"}


def main() -> None:
    rows: list[dict] = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

    for genre_dir in DATA_DIR.iterdir():
        if not genre_dir.is_dir():
            continue

        label = genre_dir.name

        for audio_file in genre_dir.rglob("*"):
            if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            try:
                features = extract_features(audio_file)
                features["label"] = label
                features["filename"] = str(audio_file)
                rows.append(features)
                print(f"Processed: {audio_file}")
            except Exception as e:
                print(f"Skipping {audio_file}: {e}")

    if not rows:
        raise RuntimeError("No audio files were processed.")

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved dataset to {OUTPUT_CSV}")
    print(df.head())
    print(f"\nTotal samples: {len(df)}")


if __name__ == "__main__":
    main()