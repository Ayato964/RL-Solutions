"""Group Relative Policy Optimization (GRPO) Algorithm.

A Critic-free reinforcement learning algorithm originally introduced by DeepSeek,
adapted here for Gymnasium environments and grid-world games.

Key architectural properties:
1. No Critic / Value network: Cuts memory and compute in half.
2. Group-relative baseline: Computes advantages by normalizing returns across a group
   of G parallel/sampled rollouts:
       A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
3. PPO-clip objective with KL penalty against a reference policy.
"""
from copy import deepcopy
import os
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymnasium import Env, spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_game.rl.config import GRPOConfig, RLConfig
from rl_game.rl.features_extractor import get_features_extractor_specs


class GRPOActor(nn.Module):
    """Critic-free Actor network for GRPO supporting both step-by-step and full-episode sequence modeling."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_dim: int,
        features_extractor_class: Type[BaseFeaturesExtractor],
        features_extractor_kwargs: Dict[str, Any],
        net_arch: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()
        self.features_extractor = features_extractor_class(
            observation_space,
            **features_extractor_kwargs,
        )
        features_dim = self.features_extractor.features_dim

        # Action policy MLP head
        layers: List[nn.Module] = []
        last_dim = features_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.GELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, action_dim))

        self.policy_head = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> Categorical:
        """Returns the action distribution for standard observations."""
        features = self.features_extractor(observations)
        logits = self.policy_head(features)
        return Categorical(logits=logits)

    def forward_trajectory(self, trajectory_frames: torch.Tensor) -> Categorical:
        """Processes an ENTIRE episode sequence of frames (L, C, H, W) through Causal Transformer
        and returns the Categorical action distribution for ALL L steps simultaneously."""
        if hasattr(self.features_extractor, "forward_trajectory"):
            features_seq = self.features_extractor.forward_trajectory(trajectory_frames)
            if features_seq.ndim == 3:
                features_seq = features_seq.squeeze(0)
            logits = self.policy_head(features_seq)
            return Categorical(logits=logits)
        else:
            return self(trajectory_frames)

    def step_inference(
        self,
        obs_tensor: torch.Tensor,
        token_history: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step-by-step causal inference using token history cache."""
        if (
            token_history is not None
            and hasattr(self.features_extractor, "encode_frame")
            and hasattr(self.features_extractor, "forward_tokens")
        ):
            # 1. Encode ONLY the current new frame into a soft token
            new_token = self.features_extractor.encode_frame(obs_tensor)
            token_history.append(new_token)

            # 2. Concatenate all tokens from game start (s_0) to now (s_t)
            tokens = torch.cat(token_history, dim=1)

            # 3. Apply Causal Self-Attention over the entire game history
            features = self.features_extractor.forward_tokens(tokens, return_all_steps=False)
            logits = self.policy_head(features)
            dist = Categorical(logits=logits)
        else:
            dist = self(obs_tensor)

        if deterministic:
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_action_and_log_prob(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self(observations)
        if deterministic:
            action = dist.logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class GRPO:
    """Critic-Free Group Relative Policy Optimization (GRPO) Trainer.

    Conforms to the Stable-Baselines3 algorithm interface (learn, predict, save, load).
    """

    def __init__(
        self,
        env: Env,
        config: Optional[RLConfig] = None,
        learning_rate: float = 3e-4,
        group_size: int = 32,
        rollout_steps: Optional[int] = None,
        batch_size: int = 64,
        n_epochs: int = 1,
        gradient_accumulation_steps: Optional[int] = None,
        advantage_mode: str = "step_discounted",
        clip_range: float = 0.2,
        kl_coef: float = 0.0,
        ent_coef: float = 0.01,
        gamma: float = 0.99,
        max_grad_norm: float = 0.5,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        verbose: int = 1,
    ):
        self.env = env
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Extract hyperparameters
        if config and hasattr(config, "grpo"):
            grpo_cfg = config.grpo
            self.learning_rate = grpo_cfg.learning_rate
            self.group_size = grpo_cfg.group_size
            self.rollout_steps = grpo_cfg.rollout_steps
            self.batch_size = grpo_cfg.batch_size
            self.n_epochs = grpo_cfg.n_epochs
            self.gradient_accumulation_steps = getattr(grpo_cfg, "gradient_accumulation_steps", None)
            self.advantage_mode = getattr(grpo_cfg, "advantage_mode", "step_discounted")
            self.clip_range = grpo_cfg.clip_range
            self.kl_coef = grpo_cfg.kl_coef
            self.ent_coef = grpo_cfg.ent_coef
            self.gamma = grpo_cfg.gamma
            self.max_grad_norm = grpo_cfg.max_grad_norm
            self.verbose = grpo_cfg.verbose
        else:
            self.learning_rate = learning_rate
            self.group_size = group_size
            self.rollout_steps = rollout_steps
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.gradient_accumulation_steps = gradient_accumulation_steps
            self.advantage_mode = advantage_mode
            self.clip_range = clip_range
            self.kl_coef = kl_coef
            self.ent_coef = ent_coef
            self.gamma = gamma
            self.max_grad_norm = max_grad_norm
            self.verbose = verbose

        # Policy kwargs & Feature extractor
        self.policy_kwargs = dict(policy_kwargs or {})
        if "features_extractor_class" in self.policy_kwargs:
            extractor_cls = self.policy_kwargs["features_extractor_class"]
            extractor_kwargs = self.policy_kwargs.get("features_extractor_kwargs", {})
        elif config:
            extractor_cls, extractor_kwargs = get_features_extractor_specs(
                temporal_config=config.temporal,
                raw_channels=config.max_id,
            )
        else:
            from rl_game.rl.features_extractor import TemporalStackedFeaturesExtractor
            extractor_cls = TemporalStackedFeaturesExtractor
            extractor_kwargs = {"features_dim": 64}

        action_dim = env.action_space.n if isinstance(env.action_space, spaces.Discrete) else env.action_space.shape[0]

        # 1. Active Actor
        self.actor = GRPOActor(
            observation_space=env.observation_space,
            action_dim=action_dim,
            features_extractor_class=extractor_cls,
            features_extractor_kwargs=extractor_kwargs,
        ).to(self.device)

        # 2. Reference Actor (Frozen copy for KL penalty constraint)
        self.ref_actor = deepcopy(self.actor).to(self.device)
        for param in self.ref_actor.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.num_timesteps = 0

    def predict(
        self,
        observation: np.ndarray,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predicts action for a given single observation."""
        self.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(observation, device=self.device)
            if obs_tensor.ndim == 3:
                obs_tensor = obs_tensor.unsqueeze(0)
            action, _ = self.actor.get_action_and_log_prob(obs_tensor, deterministic=deterministic)
            action_np = action.cpu().numpy()
            if len(action_np) == 1:
                action_np = action_np[0]
            return action_np, None

    def _set_game_status(self, status: str) -> None:
        """Sets status string on the game score_board for UI display if supported."""
        try:
            unwrapped = getattr(self.env, "unwrapped", self.env)
            game = getattr(unwrapped, "game", None)
            if game is not None and hasattr(game, "field") and hasattr(game.field, "score_board"):
                with game.field.lock:
                    game.field.score_board["status"] = status
        except Exception:
            pass

    def _collect_group_rollouts(
        self,
        callback: Optional[BaseCallback] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], int, bool]:
        """Collects completed episode(s) by running pure inference until game-over (terminated)
        or time-over (truncated), packaging the transitions for all-at-once training."""
        self.actor.eval()
        group_trajectories: List[Dict[str, Any]] = []
        total_steps_collected = 0
        continue_training = True

        for ep_idx in range(self.group_size):
            self._set_game_status(f"INFERENCE [{ep_idx + 1}/{self.group_size}]")
            obs, _ = self.env.reset()
            t_obs: List[np.ndarray] = []
            t_actions: List[int] = []
            t_log_probs: List[float] = []
            t_rewards: List[float] = []

            step = 0
            token_history: List[torch.Tensor] = []
            while True:
                step += 1
                obs_tensor = torch.as_tensor(obs, device=self.device)
                if obs_tensor.ndim == 3:
                    obs_tensor = obs_tensor.unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor = self.actor.step_inference(
                        obs_tensor,
                        token_history=token_history,
                    )

                action = int(action_tensor.item())
                log_prob = float(log_prob_tensor.item())

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                total_steps_collected += 1
                self.num_timesteps += 1

                t_obs.append(obs)
                t_actions.append(action)
                t_log_probs.append(log_prob)
                t_rewards.append(reward)

                obs = next_obs

                if callback is not None:
                    if not callback.on_step():
                        # Signal to abort early
                        continue_training = False
                        break

                # タイムオーバー (truncated) または ゲームオーバー (terminated) になるまで推論をし続ける
                if terminated or truncated:
                    break

                if self.rollout_steps is not None and step >= self.rollout_steps:
                    break

            if len(t_actions) > 0:
                # Calculate discounted return for this completed trajectory
                returns = []
                discounted_r = 0.0
                for r in reversed(t_rewards):
                    discounted_r = r + self.gamma * discounted_r
                    returns.insert(0, discounted_r)

                trajectory_total_return = sum(t_rewards)

                group_trajectories.append({
                    "obs": np.array(t_obs),
                    "actions": np.array(t_actions),
                    "old_log_probs": np.array(t_log_probs),
                    "rewards": np.array(t_rewards),
                    "returns": np.array(returns, dtype=np.float32),
                    "total_return": trajectory_total_return,
                    "terminated": terminated,
                    "truncated": truncated,
                })

            if not continue_training:
                break

        # If empty (e.g. stopped before any steps)
        if not group_trajectories:
            return [], total_steps_collected, continue_training

        # --- Compute Advantages ---
        normalized_trajectories = []
        if self.advantage_mode == "step_discounted":
            # Pool all temporal returns-to-go G_{i, t} across all steps in the group
            all_returns = np.concatenate([traj["returns"] for traj in group_trajectories])
            mean_return = float(np.mean(all_returns))
            std_return = float(np.std(all_returns)) + 1e-8

            for traj in group_trajectories:
                if std_return > 1e-6:
                    step_advantages = (traj["returns"] - mean_return) / std_return
                else:
                    step_advantages = np.zeros_like(traj["returns"])

                normalized_trajectories.append({
                    "obs": torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device),
                    "actions": torch.as_tensor(traj["actions"], dtype=torch.long, device=self.device),
                    "old_log_probs": torch.as_tensor(traj["old_log_probs"], dtype=torch.float32, device=self.device),
                    "advantages": torch.as_tensor(step_advantages, dtype=torch.float32, device=self.device),
                    "terminated": traj.get("terminated", False),
                    "truncated": traj.get("truncated", False),
                })
        else:
            # "group_return": DeepSeek-style scalar advantage based on total episode return
            if len(group_trajectories) == 1:
                traj = group_trajectories[0]
                returns = traj["returns"]
                if len(returns) > 1 and np.std(returns) > 1e-6:
                    step_advantages = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
                else:
                    step_advantages = np.zeros_like(returns)

                normalized_trajectories.append({
                    "obs": torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device),
                    "actions": torch.as_tensor(traj["actions"], dtype=torch.long, device=self.device),
                    "old_log_probs": torch.as_tensor(traj["old_log_probs"], dtype=torch.float32, device=self.device),
                    "advantages": torch.as_tensor(step_advantages, dtype=torch.float32, device=self.device),
                    "terminated": traj.get("terminated", False),
                    "truncated": traj.get("truncated", False),
                })
            else:
                group_returns = np.array([traj["total_return"] for traj in group_trajectories])
                mean_return = float(np.mean(group_returns))
                std_return = float(np.std(group_returns)) + 1e-8

                for traj in group_trajectories:
                    if std_return > 1e-6:
                        group_advantage = (traj["total_return"] - mean_return) / std_return
                        step_advantages = np.full(len(traj["actions"]), group_advantage, dtype=np.float32)
                    else:
                        returns = traj["returns"]
                        step_advantages = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

                    normalized_trajectories.append({
                        "obs": torch.as_tensor(traj["obs"], dtype=torch.float32, device=self.device),
                        "actions": torch.as_tensor(traj["actions"], dtype=torch.long, device=self.device),
                        "old_log_probs": torch.as_tensor(traj["old_log_probs"], dtype=torch.float32, device=self.device),
                        "advantages": torch.as_tensor(step_advantages, dtype=torch.float32, device=self.device),
                        "terminated": traj.get("terminated", False),
                        "truncated": traj.get("truncated", False),
                    })

        return normalized_trajectories, total_steps_collected, continue_training

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 1,
    ) -> "GRPO":
        """Executes the Critic-free GRPO training loop.
        Infuses pure inference during gameplay, then performs all-at-once training when the episode ends.
        """
        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start(locals(), globals())

        iteration = 0
        while self.num_timesteps < total_timesteps:
            iteration += 1

            # 1. タイムオーバーやゲームオーバーになるまで推論を行い、完了したエピソードを収集
            trajectories, steps_collected, continue_training = self._collect_group_rollouts(callback=callback)

            if not continue_training:
                break

            if not trajectories:
                continue

            # 2. ゲーム終了後、収集した32本の全エピソードシーケンスを用いて一括学習 (Full-Episode Causal Sequence Training)
            self.actor.train()
            num_trajectories = len(trajectories)
            total_samples = sum(len(t["actions"]) for t in trajectories)
            if num_trajectories == 0 or total_samples == 0:
                continue

            has_trajectory_mode = hasattr(self.actor, "forward_trajectory") and hasattr(
                self.actor.features_extractor, "forward_trajectory"
            )
            policy_loss_val = 0.0

            for epoch in range(self.n_epochs):
                if self.n_epochs > 1:
                    self._set_game_status(f"TRAIN [Epoch {epoch + 1}/{self.n_epochs}]")
                else:
                    self._set_game_status("TRAIN [Updating Policy]")

                if has_trajectory_mode:
                    # Full-Episode Causal Sequence Training across all 32 trajectories
                    self.optimizer.zero_grad()
                    total_policy_loss = 0.0

                    for traj in trajectories:
                        traj_obs = torch.as_tensor(traj["obs"], device=self.device)  # (L, C, H, W)
                        traj_actions = torch.as_tensor(traj["actions"], device=self.device)  # (L,)
                        traj_old_log_probs = torch.as_tensor(traj["old_log_probs"], device=self.device)  # (L,)
                        traj_adv = traj["advantages"].to(self.device)  # (L,)

                        # Forward entire episode sequence through Causal Transformer in one parallel pass
                        dist = self.actor.forward_trajectory(traj_obs)
                        new_log_probs = dist.log_prob(traj_actions)
                        entropy = dist.entropy().mean()

                        # Probability ratio: r(theta) = pi_theta / pi_old
                        ratio = torch.exp(new_log_probs - traj_old_log_probs)

                        # Clipped surrogate objective
                        surr1 = ratio * traj_adv
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * traj_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        # Exact analytical Categorical KL divergence against reference policy
                        if self.kl_coef > 0:
                            with torch.no_grad():
                                ref_dist = self.ref_actor.forward_trajectory(traj_obs)
                            raw_kl = torch.distributions.kl.kl_divergence(dist, ref_dist)
                            kl_penalty = torch.clamp(raw_kl, min=0.0).mean()
                            kl_loss = self.kl_coef * kl_penalty
                        else:
                            kl_loss = torch.tensor(0.0, device=self.device)

                        total_loss = policy_loss + kl_loss - self.ent_coef * entropy

                        # 累積勾配 (Gradient Accumulation): 32エピソードで除算して累積
                        loss_scaled = total_loss / num_trajectories
                        loss_scaled.backward()
                        total_policy_loss += policy_loss.item()

                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    policy_loss_val = total_policy_loss / num_trajectories
                else:
                    # Fallback mini-batch training for non-sequence feature extractors
                    b_obs = torch.cat([t["obs"] for t in trajectories], dim=0)
                    b_actions = torch.cat([t["actions"] for t in trajectories], dim=0)
                    b_old_log_probs = torch.cat([t["old_log_probs"] for t in trajectories], dim=0)
                    b_advantages = torch.cat([t["advantages"] for t in trajectories], dim=0)

                    permutation = torch.randperm(total_samples)
                    batch_starts = list(range(0, total_samples, self.batch_size))
                    num_batches = len(batch_starts)
                    accum_steps = (
                        min(num_batches, max(1, self.gradient_accumulation_steps))
                        if self.gradient_accumulation_steps is not None
                        else num_batches
                    )

                    self.optimizer.zero_grad()
                    for b_idx, start_idx in enumerate(batch_starts):
                        batch_indices = permutation[start_idx : start_idx + self.batch_size]

                        mb_obs = b_obs[batch_indices]
                        mb_actions = b_actions[batch_indices]
                        mb_old_log_probs = b_old_log_probs[batch_indices]
                        mb_advantages = b_advantages[batch_indices]

                        dist = self.actor(mb_obs)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()

                        if self.kl_coef > 0:
                            with torch.no_grad():
                                ref_dist = self.ref_actor(mb_obs)
                            raw_kl = torch.distributions.kl.kl_divergence(dist, ref_dist)
                            kl_penalty = torch.clamp(raw_kl, min=0.0).mean()
                            kl_loss = self.kl_coef * kl_penalty
                        else:
                            kl_loss = torch.tensor(0.0, device=self.device)

                        total_loss = policy_loss + kl_loss - self.ent_coef * entropy
                        loss_scaled = total_loss / accum_steps
                        loss_scaled.backward()

                        is_accum_boundary = ((b_idx + 1) % accum_steps == 0) or ((b_idx + 1) == num_batches)
                        if is_accum_boundary:
                            if self.max_grad_norm > 0:
                                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    policy_loss_val = policy_loss.item()

            self._set_game_status("INFERENCE")

            if self.verbose > 0 and iteration % log_interval == 0:
                epoch_str = f"{self.n_epochs} epoch" if self.n_epochs == 1 else f"{self.n_epochs} epochs"
                mode_str = "Full-Episode Sequence" if has_trajectory_mode else "MiniBatch Accumulation"
                print(
                    f"[GRPO Batch {iteration}] Timesteps: {self.num_timesteps}/{total_timesteps} "
                    f"| Episodes: {num_trajectories}/{self.group_size} | Steps: {total_samples} "
                    f"| Policy Loss: {policy_loss_val:.4f} "
                    f"| Trained ({epoch_str}, {mode_str})",
                    flush=True,
                )

        if callback is not None:
            callback.on_training_end()

        return self

    def save(self, path: str) -> None:
        """Saves model weights to disk."""
        if not path.endswith(".zip") and not path.endswith(".pt"):
            save_path = f"{path}.pt"
        else:
            save_path = path

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "num_timesteps": self.num_timesteps,
            "learning_rate": self.learning_rate,
        }, save_path)

    @classmethod
    def load(cls, path: str, env: Env, config: Optional[RLConfig] = None) -> "GRPO":
        """Loads model weights from disk."""
        load_path = path if (path.endswith(".pt") or path.endswith(".zip")) else f"{path}.pt"
        model = cls(env=env, config=config)
        checkpoint = torch.load(load_path, map_location=model.device)
        model.actor.load_state_dict(checkpoint["actor_state_dict"])
        model.num_timesteps = checkpoint.get("num_timesteps", 0)
        return model
