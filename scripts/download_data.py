
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


DATASET = "arafatsahinafridi/multi-class-driver-behavior-image-dataset"


def download_dataset(output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "No esta instalada la CLI de Kaggle. Ejecuta: pip install kaggle"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle no pudo descargar el dataset. Verifica que kaggle.json exista en "
            "%USERPROFILE%\\.kaggle\\kaggle.json y que aceptaste las condiciones del dataset."
        ) from exc

    zip_files = sorted(output_path.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not zip_files:
        raise FileNotFoundError("Kaggle no genero un archivo .zip en el directorio destino.")

    zip_path = zip_files[-1]
    extract_dir = output_path / "module2_distraction"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as zip_file:
        zip_file.extractall(extract_dir)

    return extract_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el dataset del modulo 2 desde Kaggle.")
    parser.add_argument("--output-dir", default="data/raw", help="Directorio para guardar el dataset.")
    args = parser.parse_args()
    extract_dir = download_dataset(args.output_dir)
    print(f"Dataset extraido en: {extract_dir}")


if __name__ == "__main__":
    main()
