
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "docs" / "report"

DISTRACTION_MODULE_DIR = PROJECT_ROOT / "src" / "module2_distraction"
DISTRACTION_DATASET_DIR = RAW_DATA_DIR / "multi_class_driver_behavior"
DISTRACTION_MODEL_DIR = MODELS_DIR / "distraction"
DISTRACTION_ARTIFACTS_DIR = DISTRACTION_MODEL_DIR / "artifacts"

DRIVER_BEHAVIOR_CLASSES = (
    "Safe Driving",
    "Turning",
    "Texting Phone",
    "Talking Phones",
    "Others",
)

DISTRACTION_RISK_CLASSES = (
    "Turning",
    "Texting Phone",
    "Talking Phones",
    "Others",
)

DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 2
DEFAULT_RANDOM_SEED = 42
DEFAULT_MODEL_NAME = "resnet18"
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 12
DEFAULT_PATIENCE = 4

KAGGLE_DATASET = "arafatsahinafridi/multi-class-driver-behavior-image-dataset"
