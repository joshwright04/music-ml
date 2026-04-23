import os

def clear_output_directory(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return

    for file_name in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)

    print(f"Cleared old files from {output_dir}")