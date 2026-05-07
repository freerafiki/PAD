"""
Training configuration for the ranking model.

Centralizes all hyperparameters in one place for easy experimentation.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    DATA_ROOT: str = "/media/lucap/big_data/datasets/wikiart_PAD/PAD_dataset__Wikiart"
    MAX_NEGATIVES_PER_POSITIVE: int = 12
    TRAIN_RATIO: float = 0.8
    SEED: int = 42
    DEBUG: bool = False  # Limit dataset to 1000 images per category for fast testing
    # Geometric feature computation
    RADIUS: int = 25
    THRESHOLD: int = 25


@dataclass
class ModelConfig:
    DINO_MODEL: str = "facebook/dinov2-base"
    VIT_MODEL: str = "google/vit-base-patch16-224"
    FROZEN_LAYERS: int = 8
    DROPOUT: float = 0.5
    GEOMETRIC_CHANNEL_SCALE: float = 1.0  # Multiplier for channels 3-5 of rgb_geometric


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 4
    GRAD_CLIP_MAX_NORM: float = 1.0
    BCE_POS_WEIGHT: float = 4.0
    DEVICE: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")


@dataclass
class LossConfig:
    BCE_WEIGHT: float = 0.15
    RANKING_WEIGHT: float = 0.55
    BOUNDARY_WEIGHT: float = 0.3
    RANKING_MARGIN: float = 0.3
    HARD_NEGATIVE_WEIGHT: float = 2.0
    TOP_N: int = 3
    TEMPERATURE: float = 1.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
