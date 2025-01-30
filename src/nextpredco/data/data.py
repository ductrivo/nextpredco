from pathlib import Path

DATA_DIR = Path(__file__).parent


def list_example_projects() -> list[str]:
    # List all folders in data_dir

    return [
        f.name for f in DATA_DIR.iterdir() if f.is_dir() and '__' not in f.name
    ]
