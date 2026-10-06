from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union


class AlgorithmType(str, Enum):
    """Supported Reinforcement Learning algorithms."""
    PPO = "PPO"
    DQN = "DQN"
    GRPO = "GRPO"


class TemporalType(str, Enum):
    """Supported temporal / time-series architectures."""
    LSTM = "lstm"
    GRU = "gru"
    CONV1D = "conv1d"
    STACK = "stack"
    TRANSFORMER = "transformer"


@dataclass
class TemporalConfig:
    """Configuration for temporal / time-series observation and modeling.

    Attributes:
        enabled: Whether temporal frame stacking / sequence processing is active.
        n_frames: Number of consecutive past frames in the temporal window.
        temporal_type: Type of temporal aggregator ('lstm', 'gru', 'conv1d', 'stack', 'transformer').
        hidden_dim: Hidden state / d_model dimension for recurrent and transformer layers.
        features_dim: Output dimension of the feature extractor.
        spatial_channels: Channel depths for the spatial feature extractor.
        n_heads: Number of attention heads for Transformer.
        n_layers: Number of Transformer encoder layers.
        dim_feedforward: Dimension of FeedForward layer in Transformer.
        dropout: Dropout probability.
        aggregation: Aggregation strategy for transformer tokens ('last', 'mean', 'attention').
        max_seq_len: Maximum sequence length supported by Transformer positional embeddings.
        full_episode: Whether to process the entire episode (start to game-over) as a single sequence.
    """
    enabled: bool = True
    n_frames: int = 4
    temporal_type: Union[TemporalType, str] = TemporalType.TRANSFORMER
    hidden_dim: int = 256
    features_dim: int = 256
    spatial_channels: Tuple[int, int] = (64, 128)
    n_heads: int = 4
    n_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.0
    aggregation: str = "last"
    max_seq_len: int = 1000
    full_episode: bool = True

    def __post_init__(self):
        if isinstance(self.temporal_type, str):
            self.temporal_type = TemporalType(self.temporal_type.lower())
        if self.n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {self.n_frames}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.features_dim < 1:
            raise ValueError(f"features_dim must be >= 1, got {self.features_dim}")
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})")
        if self.aggregation.lower() not in ("last", "mean", "attention"):
            raise ValueError(f"aggregation must be 'last', 'mean', or 'attention', got {self.aggregation}")


@dataclass
class PPOConfig:
    """Hyperparameters for Proximal Policy Optimization (PPO)."""
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    verbose: int = 1
    tensorboard_log: Optional[str] = "./ppo_custom_log/"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DQNConfig:
    """Hyperparameters for Deep Q-Network (DQN)."""
    learning_rate: float = 1e-4
    buffer_size: int = 50_000
    learning_starts: int = 1000
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: Union[int, Tuple[int, str]] = (4, "step")
    gradient_steps: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    verbose: int = 1
    tensorboard_log: Optional[str] = "./dqn_custom_log/"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization (GRPO).

    Eliminates the Critic / Value network entirely by evaluating actions relative
    to a group of sampled rollouts.
    """
    learning_rate: float = 3e-4
    group_size: int = 32             # G: number of completed episodes per training batch (e.g. 32 full games)
    rollout_steps: Optional[int] = None # None: infer continuously until terminated (Game Over) or truncated (Time Over)
    batch_size: int = 64
    n_epochs: int = 1                # 1 epoch per rollout group to prevent off-policy drift and ratio saturation
    gradient_accumulation_steps: Optional[int] = None # None: accumulate all mini-batches in the 32 rollouts for 1 unified update
    advantage_mode: str = "step_discounted" # 'step_discounted': temporal returns-to-go G_t, 'group_return': DeepSeek episode scalar
    clip_range: float = 0.2
    kl_coef: float = 0.0             # beta: weight for KL divergence constraint (0.0 for RL from scratch)
    ent_coef: float = 0.01           # Entropy regularization coefficient
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    verbose: int = 1
    tensorboard_log: Optional[str] = "./grpo_custom_log/"
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if self.gradient_accumulation_steps is not None and self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1 when set, got {self.gradient_accumulation_steps}")
        if self.advantage_mode not in ("step_discounted", "group_return"):
            raise ValueError(f"advantage_mode must be 'step_discounted' or 'group_return', got {self.advantage_mode}")
        if self.rollout_steps is not None and self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1 when set, got {self.rollout_steps}")



@dataclass
class ImitationConfig:
    """Hyperparameters and settings for Human Demonstration & Behavioral Cloning Pre-training.

    Attributes:
        enabled: Whether human demo collection & behavioral cloning is active before RL.
        epochs: Number of supervised training epochs over the demonstration buffer.
        batch_size: Mini-batch size for imitation learning.
        learning_rate: Optimizer learning rate for behavioral cloning.
        weight_decay: L2 regularization coefficient.
        min_demos: Minimum number of demonstration steps required to run pre-training.
        save_dir: Directory path for persisting and loading human demonstration files (.npz).
    """
    enabled: bool = True
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_demos: int = 1
    save_dir: Optional[str] = None
    max_episodes: int = 100
    min_episode_steps: int = 2


    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.min_demos < 1:
            raise ValueError(f"min_demos must be >= 1, got {self.min_demos}")
        if self.max_episodes < 1:
            raise ValueError(f"max_episodes must be >= 1, got {self.max_episodes}")
        if self.min_episode_steps < 1:
            raise ValueError(f"min_episode_steps must be >= 1, got {self.min_episode_steps}")
        if self.save_dir is not None and not isinstance(self.save_dir, str):
            raise TypeError(f"save_dir must be a string or None, got {type(self.save_dir)}")



@dataclass
class RLConfig:
    """Master Reinforcement Learning configuration object.

    Allows seamless switching between algorithms (PPO vs DQN vs GRPO) and temporal models,
    as well as configuring runtime behavior.
    """
    algorithm: Union[AlgorithmType, str] = AlgorithmType.GRPO
    action_space: int = 4
    max_id: int = 4
    step_delay: float = 0.05
    survival_reward: float = 0.5
    total_timesteps: int = 1_000_000
    model_save_path: str = "model/rl_model"
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)

    def __post_init__(self):
        if isinstance(self.algorithm, str):
            try:
                self.algorithm = AlgorithmType(self.algorithm.upper())
            except ValueError:
                pass  # Allow external algorithms registered in RLModelFactory
        if self.action_space < 1:
            raise ValueError(f"action_space must be >= 1, got {self.action_space}")
        if self.max_id < 1:
            raise ValueError(f"max_id must be >= 1, got {self.max_id}")
        if self.step_delay < 0:
            raise ValueError(f"step_delay must be >= 0, got {self.step_delay}")
        if self.survival_reward < 0:
            raise ValueError(f"survival_reward must be >= 0, got {self.survival_reward}")

