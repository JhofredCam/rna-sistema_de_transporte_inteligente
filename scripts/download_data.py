
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.config import DISTRACTION_DATASET_DIR, KAGGLE_DATASET


def download_kaggle_dataset(
    dataset: str = KAGGLE_DATASET,
    output_dir: str | Path = DISTRACTION_DATASET_DIR,
    unzip: bool = True,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not shutil.which("kaggle"):
        raise RuntimeError("Kaggle CLI is not installed. Run: pip install kaggle")

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(output_path),
    ]
    if unzip:
        command.append("--unzip")
    subprocess.run(command, check=True)

    if unzip:
        for zip_path in output_path.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(output_path)
            zip_path.unlink()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the driver behavior image dataset.")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--output-dir", default=str(DISTRACTION_DATASET_DIR))
    parser.add_argument("--no-unzip", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("KAGGLE_USERNAME") and not Path.home().joinpath(".kaggle/kaggle.json").exists():
        raise RuntimeError(
            "Kaggle credentials were not found. In Colab, upload kaggle.json or set "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )

    path = download_kaggle_dataset(args.dataset, args.output_dir, unzip=not args.no_unzip)
    print(f"Dataset available at: {path}")


if __name__ == "__main__":
    main()
