"""
Training configuration for the CNN ranking model.

Centralizes all hyperparameters in one place for easy experimentation.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    DATA_ROOT: str = "/media/lucap/big_data/datasets/wikiart_PAD/PAD_dataset__Wikiart"
    MIN_NEGATIVES_PER_POSITIVE: int = 4
    MAX_NEGATIVES_PER_POSITIVE: int = 16
    TRAIN_RATIO: float = 0.8
    SEED: int = 42
    DEBUG: bool = False
    USE_GEOMETRIC: bool = False
    NUM_IMAGES: int = 200000
    NUM_IMAGES_VAL: int = 20000
    POSITIVE_RATIO: float = 0.07
    SAME_PAIR_BATCH: bool = False
    RADIUS: int = 25
    THRESHOLD: int = 25


@dataclass
class ModelConfig:
    TYPE: str = 'puzzle'  # puzzle, gated, guidance_gated
    DROPOUT: float = 0.5
    CNN_CHANNELS: int = 64
    CNN_BLOCK_CHANNELS: tuple = (64, 128, 256)


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 30
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    EARLY_STOPPING_PATIENCE: int = 7
    GRAD_CLIP_MAX_NORM: float = 1.0
    BCE_POS_WEIGHT: float = 5.0
    DEVICE: str = field(default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu")


@dataclass
class LossConfig:
    BCE_WEIGHT: float = 1
    RANKING_WEIGHT: float = 0
    BOUNDARY_WEIGHT: float = 0
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
    name: str = "Option3_CNN_single_img"
