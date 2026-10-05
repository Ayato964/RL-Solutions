from rl_game.rl.config import (
    AlgorithmType,
    DQNConfig,
    GRPOConfig,
    PPOConfig,
    RLConfig,
    TemporalConfig,
    TemporalType,
)
from rl_game.rl.environment import GridRLEnv
from rl_game.rl.factory import (
    DQNStrategy,
    GRPOStrategy,
    PPOStrategy,
    RLAlgorithmStrategy,
    RLModelFactory,
)
from rl_game.rl.grpo import GRPO, GRPOActor
from rl_game.rl.features_extractor import (
    AIPlayer,
    TemporalConv1dFeaturesExtractor,
    TemporalGRUFeaturesExtractor,
    TemporalLSTMFeaturesExtractor,
    TemporalStackedFeaturesExtractor,
    TemporalTransformerFeaturesExtractor,
    get_features_extractor_specs,
)
from rl_game.rl.temporal import TemporalObservationBuffer

# Backward compatibility alias
DQNTrainer = GridRLEnv

__all__ = [
    "AlgorithmType",
    "TemporalType",
    "RLConfig",
    "TemporalConfig",
    "PPOConfig",
    "DQNConfig",
    "GRPOConfig",
    "GridRLEnv",
    "DQNTrainer",
    "TemporalObservationBuffer",
    "RLModelFactory",
    "RLAlgorithmStrategy",
    "PPOStrategy",
    "DQNStrategy",
    "GRPOStrategy",
    "GRPO",
    "TemporalLSTMFeaturesExtractor",
    "TemporalGRUFeaturesExtractor",
    "TemporalConv1dFeaturesExtractor",
    "TemporalStackedFeaturesExtractor",
    "TemporalTransformerFeaturesExtractor",
    "AIPlayer",
    "get_features_extractor_specs",
]
