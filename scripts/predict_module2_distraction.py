from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module2_distraction.classifier import (
    DriverDistractionClassifier,
    preventive_measure_for,
)
from src.shared.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clasifica imagenes con el modulo 2.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", default="models/module2_distraction/predictions.json")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = DriverDistractionClassifier(args.checkpoint, device=args.device)
    predictions = []
    for image in args.images:
        prediction = classifier.predict(image)
        row = prediction.__dict__
        row["preventive_measure"] = preventive_measure_for(prediction.predicted_label)
        predictions.append(row)
        print(f"{image}: {prediction.predicted_label} ({prediction.confidence:.2%})")
    save_json({"predictions": predictions}, Path(args.output))


if __name__ == "__main__":
    main()
