"""Training script for PruneVision."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

try:
    from backend.model import PrunableCNN, evaluate_accuracy, train_prunable_model
except ModuleNotFoundError:
    from model import PrunableCNN, evaluate_accuracy, train_prunable_model


LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BACKEND_DIR / "model_checkpoints"
RESULTS_PATH = BASE_DIR / "results.json"
HISTORY_PATH = BACKEND_DIR / "training_history.json"
GATE_PLOT_PATH = BASE_DIR / "gate_distribution_best.png"
PLANT_VILLAGE_ENV = "PLANTVILLAGE_DIR"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
DEFAULT_LAMBDAS = [0.0001, 0.001, 0.01]
QUICK_MODE_EPOCHS = 2
QUICK_MODE_MAX_SAMPLES = 2000
QUICK_MODE_LAMBDAS = [0.001]
DEFAULT_TEST_SPLIT = 0.15
DEFAULT_VAL_SPLIT = 0.15
DEFAULT_TRAIN_SPLIT = 0.70
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
PLANTVILLAGE_CANDIDATES = [
    BASE_DIR / "data" / "PlantVillage",
    BASE_DIR / "dataset" / "PlantVillage",
    BACKEND_DIR / "data" / "PlantVillage",
]


@dataclass(slots=True)
class LambdaResult:
    """Summary of one sparse training run."""

    lambda_sparse: float
    test_accuracy: float
    sparsity_percent: float
    model_path: str
    class_names: list[str]
    history: list[dict[str, Any]]


@dataclass(slots=True)
class TrainConfig:
    """Runtime configuration for the training workflow."""

    epochs: int
    batch_size: int
    num_workers: int
    lambdas: list[float]
    max_samples: Optional[int]
    quick_mode: bool


def configure_logging() -> None:
    """Configure module logging."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def parse_arguments() -> TrainConfig:
    """Parse CLI arguments for training behavior."""

    parser = argparse.ArgumentParser(description="Train PruneVision sparse models")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs per lambda run")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument(
        "--lambdas",
        type=str,
        default=",".join(str(value) for value in DEFAULT_LAMBDAS),
        help="Comma-separated lambda values, e.g. 0.0001,0.001,0.01",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on dataset size before splitting",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast CPU-friendly training profile",
    )
    args = parser.parse_args()

    lambdas = [float(value.strip()) for value in args.lambdas.split(",") if value.strip()]
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lambdas=lambdas,
        max_samples=args.max_samples,
        quick_mode=args.quick,
    )

    if config.quick_mode:
        config.epochs = QUICK_MODE_EPOCHS
        config.max_samples = QUICK_MODE_MAX_SAMPLES
        config.lambdas = QUICK_MODE_LAMBDAS
        LOGGER.info(
            "Quick mode enabled: epochs=%d max_samples=%d lambdas=%s",
            config.epochs,
            config.max_samples,
            config.lambdas,
        )

    return config


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducible splits and training."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and evaluation transforms."""

    train_transform = transforms.Compose(
        [
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD),
        ]
    )
    return train_transform, eval_transform


def locate_plant_village_root() -> Optional[Path]:
    """Locate a local PlantVillage dataset root if it exists."""

    env_value = os.environ.get(PLANT_VILLAGE_ENV)
    if env_value:
        candidate = Path(env_value)
        if candidate.exists():
            return candidate

    for candidate in PLANTVILLAGE_CANDIDATES:
        if candidate.exists():
            return candidate

    return None


def load_base_dataset() -> tuple[Dataset, list[str]]:
    """Load PlantVillage if available, otherwise fallback to CIFAR-10 or FakeData."""

    plant_village_root = locate_plant_village_root()
    if plant_village_root is not None:
        dataset = datasets.ImageFolder(plant_village_root, transform=None)
        class_names = dataset.classes
        LOGGER.info("Loaded PlantVillage dataset from %s", plant_village_root)
        return dataset, class_names

    try:
        dataset = datasets.CIFAR10(root=str(BASE_DIR / "data"), train=True, download=True, transform=None)
        class_names = list(dataset.classes)
        LOGGER.info("Loaded CIFAR-10 fallback dataset")
        return dataset, class_names
    except Exception as exc:
        LOGGER.warning("CIFAR-10 fallback failed, using FakeData: %s", exc)
        dataset = datasets.FakeData(
            size=5000,
            image_size=(3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            num_classes=10,
            transform=None,
        )
        class_names = [f"Class {index}" for index in range(10)]
        return dataset, class_names


def maybe_limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    """Optionally cap dataset size to speed up experimentation."""

    if max_samples is None or max_samples <= 0:
        return dataset

    capped_size = min(max_samples, len(dataset))
    if capped_size == len(dataset):
        return dataset

    indices = list(range(capped_size))
    LOGGER.info("Using capped dataset size: %d/%d", capped_size, len(dataset))
    return torch.utils.data.Subset(dataset, indices)


class TransformedSubset(Dataset):
    """Apply a transform to items returned by a subset."""

    def __init__(self, subset: Dataset, transform: transforms.Compose) -> None:
        """Store the subset and transform."""

        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of items in the wrapped subset."""

        return len(self.subset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Fetch an item and apply the transform before returning it."""

        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataloaders(
    dataset: Dataset,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split the dataset into train, validation, and test loaders."""

    total_size = len(dataset)
    train_size = int(total_size * DEFAULT_TRAIN_SPLIT)
    val_size = int(total_size * DEFAULT_VAL_SPLIT)
    test_size = total_size - train_size - val_size

    train_subset, val_subset, test_subset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(DEFAULT_SEED),
    )

    train_loader = DataLoader(TransformedSubset(train_subset, train_transform), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(TransformedSubset(val_subset, eval_transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(TransformedSubset(test_subset, eval_transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def evaluate_on_loader(model: PrunableCNN, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy using both PyTorch and scikit-learn metrics for verification."""

    model.eval()
    all_predictions: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            predictions = logits.argmax(dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_targets.extend(batch_targets.tolist())
    return float(accuracy_score(all_targets, all_predictions))


def plot_gate_distribution(model: PrunableCNN, output_path: Path) -> None:
    """Generate a histogram of gate values for the best model."""

    gates = model.get_all_gate_values().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.hist(gates, bins=40, color="#00ff88", alpha=0.85, edgecolor="#0d1a0d")
    plt.title("PruneVision Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_comparison_table(results: list[LambdaResult]) -> None:
    """Print the training comparison summary in a readable table."""

    header = f"{'Lambda':<12}{'Test Accuracy':<18}{'Sparsity %':<14}{'Model Path'}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.lambda_sparse:<12.4f}{result.test_accuracy:<18.4f}{result.sparsity_percent:<14.2f}{result.model_path}"
        )


def main() -> None:
    """Run the end-to-end sparse training comparison."""

    configure_logging()
    config = parse_arguments()
    set_seed(DEFAULT_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform, eval_transform = build_transforms()
    dataset, class_names = load_base_dataset()
    dataset = maybe_limit_dataset(dataset, config.max_samples)
    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        train_transform,
        eval_transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    results: list[LambdaResult] = []
    aggregate_history: dict[str, Any] = {"runs": []}
    best_result: Optional[LambdaResult] = None
    best_model_state: Optional[dict[str, Any]] = None

    for lambda_sparse in config.lambdas:
        LOGGER.info("Starting training run for lambda=%s", lambda_sparse)
        model = PrunableCNN(num_classes=len(class_names))
        checkpoint_path = ARTIFACT_DIR / f"prunevision_lambda_{str(lambda_sparse).replace('.', '_')}.pt"
        history_entries, best_state = train_prunable_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lambda_sparse=lambda_sparse,
            epochs=config.epochs,
            checkpoint_path=checkpoint_path,
        )

        if best_state:
            model.load_state_dict(best_state["model_state_dict"])

        test_accuracy = evaluate_accuracy(model, test_loader, device)
        sparsity_percent = model.get_total_sparsity()
        model_path = str(checkpoint_path)
        run_history = [asdict(entry) for entry in history_entries]

        result = LambdaResult(
            lambda_sparse=lambda_sparse,
            test_accuracy=test_accuracy,
            sparsity_percent=sparsity_percent,
            model_path=model_path,
            class_names=class_names,
            history=run_history,
        )
        results.append(result)
        aggregate_history["runs"].append(asdict(result))

        if best_result is None or result.test_accuracy > best_result.test_accuracy:
            best_result = result
            best_model_state = best_state

    if best_result is None or best_model_state is None:
        raise RuntimeError("Training did not produce a valid best model")

    best_checkpoint_path = Path(best_result.model_path)
    checkpoint_payload = {
        "version": 1,
        "model_state_dict": best_model_state["model_state_dict"],
        "lambda_sparse": best_result.lambda_sparse,
        "class_names": class_names,
        "test_accuracy": best_result.test_accuracy,
        "train_history": best_result.history,
    }
    best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, best_checkpoint_path)
    (BACKEND_DIR / "model_checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload, BACKEND_DIR / "model_checkpoints" / "best_model.pt")

    aggregate_history["best_lambda"] = best_result.lambda_sparse
    aggregate_history["best_test_accuracy"] = best_result.test_accuracy
    aggregate_history["best_sparsity_percent"] = best_result.sparsity_percent
    HISTORY_PATH.write_text(json.dumps(aggregate_history, indent=2), encoding="utf-8")

    comparison_payload = {
        "results": [asdict(result) for result in results],
        "best_lambda": best_result.lambda_sparse,
        "best_test_accuracy": best_result.test_accuracy,
        "best_sparsity_percent": best_result.sparsity_percent,
        "class_names": class_names,
    }
    RESULTS_PATH.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    best_model = PrunableCNN(num_classes=len(class_names)).to(device)
    best_model.load_state_dict(best_model_state["model_state_dict"])
    plot_gate_distribution(best_model, GATE_PLOT_PATH)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
