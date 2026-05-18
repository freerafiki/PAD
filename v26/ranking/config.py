"""
Training configuration for the ranking model.

Centralizes all hyperparameters in one place for easy experimentation.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    DATA_ROOT: str = "/media/lucap/big_data/datasets/wikiart_PAD/PAD_dataset__Wikiart_nn"
    MAX_NEGATIVES_PER_POSITIVE: int = 12
    TRAIN_RATIO: float = 0.8
    SEED: int = 42
    DEBUG: bool = False  # Limit dataset to 1000 images per category for fast testing
    # Geometric feature computation
    RADIUS: int = 25
    THRESHOLD: int = 25


@dataclass
class ModelConfig:
    # DINO_MODEL: str = "facebook/dinov2-base"
    DINO_MODEL: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    VIT_MODEL: str = "google/vit-base-patch16-224"
    FROZEN_LAYERS: int = 8
    DROPOUT: float = 0.5
    GEOMETRIC_CHANNEL_SCALE: float = 1.0  # Multiplier for channels 3-5 of rgb_geometric
    FILM_ENABLED: bool = False
    FILM_T_DIM: int = 64
    FILM_LAYERS: tuple = (8, 9, 10, 11)


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
    BCE_WEIGHT: float = 0.4
    RANKING_WEIGHT: float = 0.4
    BOUNDARY_WEIGHT: float = 0.2
    RANKING_MARGIN: float = 0.3
    HARD_NEGATIVE_WEIGHT: float = 2.0
    TOP_N: int = 3
    TEMPERATURE: float = 1.0


@dataclass
class AugmentationConfig:
    ENABLED: bool = False
    COLOR_JITTER: bool = True
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1
    GAUSSIAN_BLUR: bool = True
    GAUSSIAN_BLUR_KERNEL_SIZE: int = 3
    GAUSSIAN_BLUR_PROB: float = 0.2
    RANDOM_GRAYSCALE: bool = True
    RANDOM_GRAYSCALE_PROB: float = 0.05
    HORIZONTAL_FLIP: bool = True
    HORIZONTAL_FLIP_PROB: float = 0.5
    VERTICAL_FLIP: bool = False
    VERTICAL_FLIP_PROB: float = 0.5
    ROTATION_90: bool = True
    ROTATION_90_PROB: float = 0.3


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
