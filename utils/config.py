"""Single source of truth for paths, hyperparameters, and seeds.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data/raw"
    raw_web: Path = ROOT / "data/raw/web"
    raw_web2: Path = ROOT / "data/raw/web2"
    raw_custom: Path = ROOT / "data/raw/custom"
    processed: Path = ROOT / "data/preprocessed"
    manifest: Path = ROOT / "data/preprocessed/manifest.csv"
    crops: Path = ROOT / "data/preprocessed/crops"
    crops_manifest: Path = ROOT / "data/preprocessed/crops/crops_manifest.csv"
    splits: Path = ROOT / "data/preprocessed/splits"
    norm_stats: Path = ROOT / "data/preprocessed/normalization_stats.json"
    results: Path = ROOT / "results"
    models: Path = ROOT / "results/models"
    figures: Path = ROOT / "results/figures"
    tables: Path = ROOT / "results/tables"
    hardware: Path = ROOT / "results/hardware"
    manuscript: Path = ROOT / "manuscript"


@dataclass(frozen=True)
class DataConfig:
    web_padding: float = 0.15
    web2_padding: float = 0.05
    web3_padding: float = 0.15  # added 2026-05-15; web3 bboxes visually as tight as web1
    crop_size: int = 64
    intermediate_crop_size: int = 96
    n_folds: int = 5
    iou_noisy_threshold: float = 0.10
    # Sources to include in the bag-level pipeline.
    # `custom` has no labels yet, so it is excluded from the labeled pipeline.
    labeled_sources: tuple = ("web", "web2", "web3")


@dataclass(frozen=True)
class ModelConfig:
    n_qubits: int = 8
    n_quantum_layers: int = 3
    projection_dim: int = 8
    backbone: str = "resnet18"
    backbone_pretrained: bool = True
    backbone_freeze_until: str = "layer4"


@dataclass(frozen=True)
class TrainConfig:
    instance_pretrain_epochs: int = 80
    mil_epochs: int = 200
    instance_lr: float = 1e-4
    classical_lr: float = 1e-3
    quantum_lr: float = 5e-2
    weight_decay: float = 1e-4
    batch_bags: int = 1
    accumulation_steps: int = 4
    early_stop_patience: int = 30
    count_loss_weight: float = 0.5
    seed: int = 42


@dataclass(frozen=True)
class HardwareConfig:
    primary_backend: str = "ibm_boston"
    secondary_backend: str = "ibm_miami"
    shots: int = 4096
    resilience_level: int = 2
    optimization_level: int = 3
    n_pauli_twirls: int = 8


PATHS = Paths()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
HARDWARE = HardwareConfig()


def ensure_dirs() -> None:
    """Create all output directories. Idempotent."""
    for p in (
        PATHS.processed,
        PATHS.crops,
        PATHS.splits,
        PATHS.results,
        PATHS.models,
        PATHS.figures,
        PATHS.tables,
        PATHS.hardware,
    ):
        p.mkdir(parents=True, exist_ok=True)
