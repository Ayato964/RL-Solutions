from rl_game.rl.config import (
    AlgorithmType,
    DQNConfig,
    GRPOConfig,
    ImitationConfig,
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
    SpatialCNNEncoder,
    TemporalConv1dFeaturesExtractor,
    TemporalGRUFeaturesExtractor,
    TemporalLSTMFeaturesExtractor,
    TemporalStackedFeaturesExtractor,
    TemporalTransformerFeaturesExtractor,
    get_features_extractor_specs,
)
from rl_game.rl.temporal import TemporalObservationBuffer
from rl_game.rl.imitation import (
    BaseImitationLearner,
    DemonstrationBuffer,
    DQNImitationLearner,
    GRPOImitationLearner,
    ImitationLearnerFactory,
    PPOImitationLearner,
)

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
    "ImitationConfig",
    "GridRLEnv",
    "DQNTrainer",
    "TemporalObservationBuffer",
    "RLModelFactory",
    "RLAlgorithmStrategy",
    "PPOStrategy",
    "DQNStrategy",
    "GRPOStrategy",
    "GRPO",
    "GRPOActor",
    "TemporalLSTMFeaturesExtractor",
    "TemporalGRUFeaturesExtractor",
    "TemporalConv1dFeaturesExtractor",
    "TemporalStackedFeaturesExtractor",
    "TemporalTransformerFeaturesExtractor",
    "SpatialCNNEncoder",
    "AIPlayer",
    "get_features_extractor_specs",
    "DemonstrationBuffer",
    "BaseImitationLearner",
    "GRPOImitationLearner",
    "PPOImitationLearner",
    "DQNImitationLearner",
    "ImitationLearnerFactory",
]

