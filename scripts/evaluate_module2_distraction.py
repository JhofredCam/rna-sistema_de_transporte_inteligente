from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module2_distraction.data_loader import DistractionDataConfig, create_data_loaders
from src.module2_distraction.evaluator import DistractionEvaluator
from src.module2_distraction.model import build_model
from src.shared.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua un checkpoint del modulo 2.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="models/module2_distraction/evaluation_external")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    metadata = checkpoint["metadata"]
    data = create_data_loaders(
        DistractionDataConfig(
            data_dir=args.data_dir,
            image_size=int(metadata.get("image_size", 224)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )
    model = build_model(
        num_classes=len(metadata["class_names"]),
        architecture=metadata.get("architecture", "resnet18"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = DistractionEvaluator(
        model=model,
        data_loader=data.test_loader,
        class_names=metadata["class_names"],
        device=device,
        output_dir=args.output_dir,
    ).evaluate()
    save_json(result.to_dict(), Path(args.output_dir) / "evaluation_result.json")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"F1-score: {result.metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
