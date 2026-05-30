from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.module2_distraction import DistractionTrainingConfig, train_distraction_model
from src.shared.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el modulo 2 de conduccion distractiva.")
    parser.add_argument("--data-dir", required=True, help="Ruta del dataset en formato ImageFolder.")
    parser.add_argument("--output-dir", default="models/module2_distraction")
    parser.add_argument("--architecture", default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--device", default=None, help="cpu, cuda o vacio para autodetectar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DistractionTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        architecture=args.architecture,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        device=args.device,
    )
    result = train_distraction_model(config)
    save_json(result.to_dict(), Path(args.output_dir) / "training_result.json")
    print(f"Modelo guardado en: {result.best_model_path}")
    print(f"Metricas de prueba en: {result.metrics_path}")


if __name__ == "__main__":
    main()
