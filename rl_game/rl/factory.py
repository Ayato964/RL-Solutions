from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Union
from gymnasium import Env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_game.rl.config import AlgorithmType, RLConfig
from rl_game.rl.features_extractor import get_features_extractor_specs


class RLAlgorithmStrategy(ABC):
    """Abstract Strategy interface for building reinforcement learning models."""

    @abstractmethod
    def build_model(self, config: RLConfig, env: Env) -> BaseAlgorithm:
        """Constructs and returns the configured Stable-Baselines3 algorithm."""
        pass


class PPOStrategy(RLAlgorithmStrategy):
    """Strategy for configuring and constructing Proximal Policy Optimization (PPO)."""

    def build_model(self, config: RLConfig, env: Env) -> BaseAlgorithm:
        extractor_cls, extractor_kwargs = get_features_extractor_specs(
            temporal_config=config.temporal,
            raw_channels=config.max_id,
        )

        policy_kwargs = {
            "normalize_images": False,  # Crucial: Prevent 1-hot categorical observations from being divided by 255
            "features_extractor_class": extractor_cls,
            "features_extractor_kwargs": extractor_kwargs,
        }

        ppo_cfg = config.ppo
        extra_kwargs = dict(ppo_cfg.extra_kwargs)
        if "policy_kwargs" in extra_kwargs:
            policy_kwargs.update(extra_kwargs.pop("policy_kwargs"))

        return PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            n_epochs=ppo_cfg.n_epochs,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            clip_range=ppo_cfg.clip_range,
            ent_coef=ppo_cfg.ent_coef,
            vf_coef=ppo_cfg.vf_coef,
            max_grad_norm=ppo_cfg.max_grad_norm,
            verbose=ppo_cfg.verbose,
            tensorboard_log=ppo_cfg.tensorboard_log,
            policy_kwargs=policy_kwargs,
            **extra_kwargs,
        )


class DQNStrategy(RLAlgorithmStrategy):
    """Strategy for configuring and constructing Deep Q-Networks (DQN)."""

    def build_model(self, config: RLConfig, env: Env) -> BaseAlgorithm:
        extractor_cls, extractor_kwargs = get_features_extractor_specs(
            temporal_config=config.temporal,
            raw_channels=config.max_id,
        )

        policy_kwargs = {
            "normalize_images": False,  # Crucial: Prevent 1-hot categorical observations from being divided by 255
            "features_extractor_class": extractor_cls,
            "features_extractor_kwargs": extractor_kwargs,
        }

        dqn_cfg = config.dqn
        extra_kwargs = dict(dqn_cfg.extra_kwargs)
        if "policy_kwargs" in extra_kwargs:
            policy_kwargs.update(extra_kwargs.pop("policy_kwargs"))

        return DQN(
            policy="CnnPolicy",
            env=env,
            learning_rate=dqn_cfg.learning_rate,
            buffer_size=dqn_cfg.buffer_size,
            learning_starts=dqn_cfg.learning_starts,
            batch_size=dqn_cfg.batch_size,
            tau=dqn_cfg.tau,
            gamma=dqn_cfg.gamma,
            train_freq=dqn_cfg.train_freq,
            gradient_steps=dqn_cfg.gradient_steps,
            target_update_interval=dqn_cfg.target_update_interval,
            exploration_fraction=dqn_cfg.exploration_fraction,
            exploration_initial_eps=dqn_cfg.exploration_initial_eps,
            exploration_final_eps=dqn_cfg.exploration_final_eps,
            verbose=dqn_cfg.verbose,
            tensorboard_log=dqn_cfg.tensorboard_log,
            policy_kwargs=policy_kwargs,
            **extra_kwargs,
        )


class GRPOStrategy(RLAlgorithmStrategy):
    """Strategy for configuring and constructing Critic-Free GRPO."""

    def build_model(self, config: RLConfig, env: Env) -> Any:
        extractor_cls, extractor_kwargs = get_features_extractor_specs(
            temporal_config=config.temporal,
            raw_channels=config.max_id,
        )

        policy_kwargs = {
            "features_extractor_class": extractor_cls,
            "features_extractor_kwargs": extractor_kwargs,
        }

        extra_kwargs = dict(config.grpo.extra_kwargs)
        if "policy_kwargs" in extra_kwargs:
            policy_kwargs.update(extra_kwargs.pop("policy_kwargs"))

        from rl_game.rl.grpo import GRPO
        return GRPO(
            env=env,
            config=config,
            policy_kwargs=policy_kwargs,
            **extra_kwargs,
        )


class RLModelFactory:
    """Factory responsible for instantiating RL models using registered strategies."""

    _strategies: Dict[Any, RLAlgorithmStrategy] = {}

    @classmethod
    def register(cls, algo_type: Any, strategy: RLAlgorithmStrategy) -> None:
        """Registers a new algorithm strategy, enabling seamless OCP expansion."""
        cls._strategies[algo_type] = strategy
        if hasattr(algo_type, "value"):
            cls._strategies[algo_type.value] = strategy
        elif isinstance(algo_type, str):
            cls._strategies[algo_type.upper()] = strategy

    @classmethod
    def create(cls, config: RLConfig, env: Env) -> BaseAlgorithm:
        """Instantiates an RL model matching the algorithm specified in RLConfig."""
        strategy = cls._strategies.get(config.algorithm)
        if not strategy and hasattr(config.algorithm, "value"):
            strategy = cls._strategies.get(config.algorithm.value)
        if not strategy and isinstance(config.algorithm, str):
            strategy = cls._strategies.get(config.algorithm.upper())

        if not strategy:
            supported = list(cls._strategies.keys())
            raise ValueError(
                f"Unsupported algorithm '{config.algorithm}'. Registered algorithms: {supported}"
            )
        return strategy.build_model(config, env)


# Auto-register core algorithms
RLModelFactory.register(AlgorithmType.PPO, PPOStrategy())
RLModelFactory.register(AlgorithmType.DQN, DQNStrategy())
RLModelFactory.register(AlgorithmType.GRPO, GRPOStrategy())
