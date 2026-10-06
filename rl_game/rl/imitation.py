"""Imitation Learning & Behavioral Cloning (BC) Pre-training Framework.

Provides demonstration buffering and supervised policy pre-training for
GRPO, PPO, and DQN algorithms prior to reinforcement learning.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, Union
import uuid

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_game.rl.config import AlgorithmType, ImitationConfig, RLConfig



class DemonstrationBuffer:
    """Thread-safe buffer storing human demonstration trajectories.

    Supports both step-level mini-batches and full-episode sequence trajectories
    required by Causal Transformers, with atomic disk persistence and cross-session merging.
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        max_episodes: int = 100,
        min_episode_steps: int = 2,
    ):
        self._lock = threading.RLock()
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.min_episode_steps = min_episode_steps
        self._current_episode_obs: List[np.ndarray] = []
        self._current_episode_actions: List[int] = []
        self._current_episode_rewards: List[float] = []

        # Completed episodes: list of tuples (obs_seq, action_seq, reward_seq)
        self.episodes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._total_steps = 0
        self._saved_episode_indices: set = set()
        self._loaded_files: set = set()

        if self.save_dir:
            self._cleanup_orphaned_tmp_files(self.save_dir)
            self.load_from_directory(self.save_dir)

    def _cleanup_orphaned_tmp_files(self, target_dir: str) -> None:
        """Removes orphaned temporary .npz files left behind by prior crashes or abrupt termination."""
        if not os.path.exists(target_dir):
            return
        try:
            for fname in os.listdir(target_dir):
                if fname.startswith("tmp_") and fname.endswith(".npz"):
                    tmp_file = os.path.join(target_dir, fname)
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass
        except Exception as e:
            print(f"[DEMO PERSISTENCE WARNING] Failed cleaning orphaned files: {e}", file=sys.stderr, flush=True)

    def _save_episode_file(
        self,
        obs_arr: np.ndarray,
        actions_arr: np.ndarray,
        rewards_arr: np.ndarray,
        target_dir: str,
    ) -> str:
        """Atomically saves a single episode as compressed .npz archive using temp file replacement.
        Includes retry logic with exponential backoff to handle Windows antivirus/search indexer locks."""
        os.makedirs(target_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ns_part = f"{time.time_ns():020d}"
        unique_id = uuid.uuid4().hex[:6]
        filename = f"demo_{timestamp}_{ns_part}_{unique_id}_{len(actions_arr)}steps.npz"
        final_path = os.path.abspath(os.path.join(target_dir, filename))
        tmp_path = os.path.abspath(os.path.join(target_dir, f"tmp_{unique_id}_{filename}"))

        np.savez_compressed(tmp_path, obs=obs_arr, actions=actions_arr, rewards=rewards_arr)

        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                os.replace(tmp_path, final_path)
                break
            except PermissionError as pe:
                if attempt == max_retries - 1:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                    raise pe
                time.sleep(base_delay * (2 ** attempt))
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise
        return final_path

    def add_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        """Records a single step transition."""
        with self._lock:
            self._current_episode_obs.append(np.array(obs, copy=True))
            self._current_episode_actions.append(int(action))
            self._current_episode_rewards.append(float(reward))
            self._total_steps += 1

            if terminated or truncated:
                self.finish_episode()

    def finish_episode(self, auto_save: bool = True) -> Optional[str]:
        """Finalizes the active episode and commits it to the buffer.
        Discards sub-minimal fragments shorter than min_episode_steps to prevent dataset contamination.
        If save_dir is configured and auto_save is True, saves immediately to disk."""
        with self._lock:
            if len(self._current_episode_obs) > 0:
                if len(self._current_episode_obs) < self.min_episode_steps:
                    self._current_episode_obs.clear()
                    self._current_episode_actions.clear()
                    self._current_episode_rewards.clear()
                    return None

                obs_arr = np.stack(self._current_episode_obs, axis=0)  # (L, C, H, W)
                actions_arr = np.array(self._current_episode_actions, dtype=np.int64)
                rewards_arr = np.array(self._current_episode_rewards, dtype=np.float32)
                ep_idx = len(self.episodes)
                self.episodes.append((obs_arr, actions_arr, rewards_arr))
                self._current_episode_obs.clear()
                self._current_episode_actions.clear()
                self._current_episode_rewards.clear()

                saved_path = None
                if auto_save and self.save_dir:
                    try:
                        saved_path = self._save_episode_file(obs_arr, actions_arr, rewards_arr, self.save_dir)
                        self._saved_episode_indices.add(ep_idx)
                        self._loaded_files.add(saved_path)
                    except Exception as err:
                        print(f"[DEMO PERSISTENCE WARNING] Failed to auto-save episode: {err}", file=sys.stderr, flush=True)
                return saved_path
            return None

    def save_to_directory(self, target_dir: Optional[str] = None) -> int:
        """Saves all unpersisted episodes to the target directory. Returns number of newly saved files."""
        with self._lock:
            dir_to_use = target_dir or self.save_dir
            if not dir_to_use:
                return 0
            if len(self._current_episode_obs) >= self.min_episode_steps:
                self.finish_episode(auto_save=False)
            elif len(self._current_episode_obs) > 0:
                self._current_episode_obs.clear()
                self._current_episode_actions.clear()
                self._current_episode_rewards.clear()

            newly_saved = 0
            for idx, (obs_arr, actions_arr, rewards_arr) in enumerate(self.episodes):
                if idx not in self._saved_episode_indices:
                    if len(actions_arr) < self.min_episode_steps:
                        continue
                    try:
                        saved_path = self._save_episode_file(obs_arr, actions_arr, rewards_arr, dir_to_use)
                        self._saved_episode_indices.add(idx)
                        self._loaded_files.add(saved_path)
                        newly_saved += 1
                    except Exception as err:
                        print(f"[DEMO PERSISTENCE WARNING] Failed saving episode {idx}: {err}", file=sys.stderr, flush=True)
            return newly_saved

    def load_from_directory(
        self,
        load_dir: str,
        max_episodes: Optional[int] = None,
        min_steps: Optional[int] = None,
    ) -> int:
        """Loads and merges *.npz demonstration files from load_dir.
        Enforces shape validation, step range, and maximum retention bounds.
        Returns number of newly loaded episodes."""
        with self._lock:
            if not os.path.exists(load_dir):
                return 0

            max_eps = max_episodes if max_episodes is not None else self.max_episodes
            min_st = min_steps if min_steps is not None else self.min_episode_steps

            all_files = sorted(f for f in os.listdir(load_dir) if f.startswith("demo_") and f.endswith(".npz"))
            # Bound retention to most recent demonstrations to prevent memory explosion
            files = all_files[-max_eps:] if max_eps and len(all_files) > max_eps else all_files

            loaded_count = 0
            for fname in files:
                fpath = os.path.abspath(os.path.join(load_dir, fname))
                if fpath in self._loaded_files:
                    continue

                try:
                    with np.load(fpath) as data:
                        if "obs" not in data or "actions" not in data:
                            print(f"[DEMO PERSISTENCE WARNING] File {fname} missing required keys ('obs', 'actions'). Skipping.", file=sys.stderr, flush=True)
                            continue

                        obs_arr = data["obs"]
                        actions_arr = data["actions"]
                        rewards_arr = data["rewards"] if "rewards" in data else np.zeros(len(actions_arr), dtype=np.float32)

                        # 1. Length consistency check
                        if len(obs_arr) != len(actions_arr):
                            print(f"[DEMO PERSISTENCE WARNING] Length mismatch in {fname}: obs={len(obs_arr)}, actions={len(actions_arr)}. Skipping.", file=sys.stderr, flush=True)
                            continue

                        # 2. Min steps per episode check
                        if len(actions_arr) < min_st:
                            print(f"[DEMO PERSISTENCE WARNING] Demonstration {fname} has too few steps ({len(actions_arr)} < {min_st}). Skipping.", file=sys.stderr, flush=True)
                            continue

                        # 3. Action type and value bounds check
                        if not np.issubdtype(actions_arr.dtype, np.integer) or (actions_arr < 0).any() or (actions_arr > 32).any():
                            print(f"[DEMO PERSISTENCE WARNING] Invalid action values or type in {fname}. Skipping.", file=sys.stderr, flush=True)
                            continue

                        # 4. Observation shape consistency check
                        if self.episodes:
                            expected_shape = self.episodes[0][0].shape[1:]
                            if obs_arr.shape[1:] != expected_shape:
                                print(f"[DEMO PERSISTENCE WARNING] Observation shape mismatch in {fname}: expected {expected_shape}, got {obs_arr.shape[1:]}. Skipping.", file=sys.stderr, flush=True)
                                continue

                        ep_idx = len(self.episodes)
                        self.episodes.append((obs_arr, actions_arr, rewards_arr))
                        self._total_steps += len(actions_arr)
                        self._saved_episode_indices.add(ep_idx)
                        self._loaded_files.add(fpath)
                        loaded_count += 1
                except Exception as err:
                    print(f"[DEMO PERSISTENCE WARNING] Corrupted or invalid demonstration file {fname}: {err}. Skipping.", file=sys.stderr, flush=True)

            if loaded_count > 0:
                print(f"[DEMO PERSISTENCE] Loaded and merged {loaded_count} past demonstration episodes ({self._total_steps} total steps) from '{load_dir}'.", flush=True)
            return loaded_count

    @property
    def total_steps(self) -> int:
        with self._lock:
            return self._total_steps

    @property
    def total_episodes(self) -> int:
        with self._lock:
            return len(self.episodes)

    def is_empty(self) -> bool:
        with self._lock:
            return self._total_steps == 0

    def clear(self) -> None:
        """Clears all stored in-memory demonstrations."""
        with self._lock:
            self._current_episode_obs.clear()
            self._current_episode_actions.clear()
            self._current_episode_rewards.clear()
            self.episodes.clear()
            self._total_steps = 0
            self._saved_episode_indices.clear()
            self._loaded_files.clear()

    def get_trajectories(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Returns completed episodes as a list of (observations_seq, actions_seq).

        If an active episode is unfinished, it is committed first.
        """
        with self._lock:
            if len(self._current_episode_obs) > 0:
                self.finish_episode()
            return [(ep[0], ep[1]) for ep in self.episodes]

    def get_flat_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Flattens all episodes into (all_obs, all_actions) arrays."""
        with self._lock:
            if len(self._current_episode_obs) > 0:
                self.finish_episode()
            if not self.episodes:
                raise ValueError("DemonstrationBuffer is empty. Cannot extract flat dataset.")
            all_obs = np.concatenate([ep[0] for ep in self.episodes], axis=0)
            all_actions = np.concatenate([ep[1] for ep in self.episodes], axis=0)
            return all_obs, all_actions



class BaseImitationLearner(ABC):
    """Abstract Strategy interface for behavioral cloning and imitation learning."""

    @abstractmethod
    def train(
        self,
        model: Any,
        demo_buffer: DemonstrationBuffer,
        config: ImitationConfig,
        epoch_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict[str, float]:
        """Pre-trains model parameters on expert demonstrations.

        Args:
            model: Reinforcement learning model instance.
            demo_buffer: Stored expert demonstrations.
            config: Imitation learning hyperparameters.
            epoch_callback: Optional callback(current_epoch, total_epochs, loss, accuracy)
                            for UI progress updates and event pumping.

        Returns:
            Dictionary containing metrics such as final 'loss' and 'accuracy'.
        """
        pass


class GRPOImitationLearner(BaseImitationLearner):
    """Behavioral cloning strategy for Critic-Free GRPO models.

    Supports both full-episode sequence training via Causal Transformer
    and standard step-by-step training.
    """

    def train(
        self,
        model: Any,
        demo_buffer: DemonstrationBuffer,
        config: ImitationConfig,
        epoch_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict[str, float]:
        trajectories = demo_buffer.get_trajectories()
        if not trajectories:
            return {"loss": 0.0, "accuracy": 0.0, "steps": 0}

        device = getattr(model, "device", torch.device("cpu"))
        actor = getattr(model, "actor", None)
        if actor is None:
            raise AttributeError("GRPO model has no 'actor' module.")

        actor.train()
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        is_transformer_seq = (
            hasattr(actor.features_extractor, "forward_trajectory")
            and getattr(getattr(model, "config", None), "temporal", None) is not None
            and getattr(model.config.temporal, "full_episode", True)
        )

        # Max sequence length boundary for Causal Transformer positional embeddings
        max_seq_len = getattr(actor.features_extractor, "max_seq_len", 1000)

        total_steps = sum(len(a) for _, a in trajectories)
        last_loss = 0.0
        last_acc = 0.0
        last_pump_time = time.time()

        for epoch in range(config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_count = 0

            if is_transformer_seq:
                # Sequence-level Causal Transformer training
                indices = np.random.permutation(len(trajectories))
                for idx in indices:
                    obs_seq, actions_seq = trajectories[idx]
                    # Enforce positional embedding boundaries to prevent runtime dimension mismatch
                    if len(obs_seq) > max_seq_len:
                        obs_seq = obs_seq[:max_seq_len]
                        actions_seq = actions_seq[:max_seq_len]

                    frames_t = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
                    actions_t = torch.as_tensor(actions_seq, dtype=torch.int64, device=device)

                    optimizer.zero_grad()
                    dist = actor.forward_trajectory(frames_t)
                    logits = dist.logits  # (seq_len, action_dim)
                    loss = F.cross_entropy(logits, actions_t)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer.step()

                    preds = logits.argmax(dim=-1)
                    epoch_correct += (preds == actions_t).sum().item()
                    epoch_loss += loss.item() * len(actions_seq)
                    epoch_count += len(actions_seq)

                    # Sub-second event pump prevents Windows DWM 5-second "Not Responding" freeze during heavy sequence batches
                    now = time.time()
                    if now - last_pump_time > 0.5:
                        try:
                            import pygame
                            pygame.event.pump()
                        except Exception:
                            pass
                        last_pump_time = now
            else:
                # Step-level batch training (CNN / MLP / LSTM / etc.)
                all_obs, all_actions = demo_buffer.get_flat_dataset()
                num_samples = len(all_actions)
                indices = np.random.permutation(num_samples)

                for start_idx in range(0, num_samples, config.batch_size):
                    batch_idx = indices[start_idx : start_idx + config.batch_size]
                    obs_b = torch.as_tensor(all_obs[batch_idx], dtype=torch.float32, device=device)
                    actions_b = torch.as_tensor(all_actions[batch_idx], dtype=torch.int64, device=device)

                    optimizer.zero_grad()
                    dist = actor(obs_b)
                    logits = dist.logits
                    loss = F.cross_entropy(logits, actions_b)
                    loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer.step()

                    preds = logits.argmax(dim=-1)
                    epoch_correct += (preds == actions_b).sum().item()
                    epoch_loss += loss.item() * len(batch_idx)
                    epoch_count += len(batch_idx)

                    now = time.time()
                    if now - last_pump_time > 0.5:
                        try:
                            import pygame
                            pygame.event.pump()
                        except Exception:
                            pass
                        last_pump_time = now

            last_loss = epoch_loss / max(1, epoch_count)
            last_acc = epoch_correct / max(1, epoch_count)

            if epoch_callback is not None:
                epoch_callback(epoch + 1, config.epochs, float(last_loss), float(last_acc))
                last_pump_time = time.time()

        # Synchronize reference actor with newly pre-trained actor weights ONLY if accuracy is sufficient (>= 50%)
        # This prevents catastrophic policy degeneration if human demonstrations were low-quality or accidental deaths.
        if hasattr(model, "ref_actor") and model.ref_actor is not None:
            if last_acc >= 0.5:
                model.ref_actor.load_state_dict(model.actor.state_dict())
                for param in model.ref_actor.parameters():
                    param.requires_grad = False
                print(f"[PRETRAIN] Reference actor synchronized with BC policy (Accuracy: {last_acc*100:.1f}% >= 50%).", flush=True)
            else:
                print(f"[PRETRAIN WARNING] BC accuracy ({last_acc*100:.1f}%) is below 50% threshold. Preserving original ref_actor to prevent policy degeneration.", flush=True)

        actor.eval()
        # Clean up BC optimizer and free GPU VRAM to avoid OOM in subsequent RL rollouts
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": float(last_loss),
            "accuracy": float(last_acc),
            "steps": total_steps,
            "episodes": len(trajectories),
        }


class PPOImitationLearner(BaseImitationLearner):
    """Behavioral cloning strategy for Stable-Baselines3 PPO."""

    def train(
        self,
        model: Any,
        demo_buffer: DemonstrationBuffer,
        config: ImitationConfig,
        epoch_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict[str, float]:
        if demo_buffer.is_empty():
            return {"loss": 0.0, "accuracy": 0.0, "steps": 0}

        all_obs, all_actions = demo_buffer.get_flat_dataset()
        device = model.device
        policy = model.policy
        policy.train()

        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_samples = len(all_actions)
        last_loss = 0.0
        last_acc = 0.0
        last_pump_time = time.time()

        for epoch in range(config.epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_correct = 0

            for start_idx in range(0, num_samples, config.batch_size):
                batch_idx = indices[start_idx : start_idx + config.batch_size]
                obs_b = torch.as_tensor(all_obs[batch_idx], dtype=torch.float32, device=device)
                actions_b = torch.as_tensor(all_actions[batch_idx], dtype=torch.int64, device=device)

                optimizer.zero_grad()
                dist = policy.get_distribution(obs_b)
                logits = dist.distribution.logits
                loss = F.cross_entropy(logits, actions_b)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                preds = logits.argmax(dim=-1)
                epoch_correct += (preds == actions_b).sum().item()
                epoch_loss += loss.item() * len(batch_idx)

                now = time.time()
                if now - last_pump_time > 0.5:
                    try:
                        import pygame
                        pygame.event.pump()
                    except Exception:
                        pass
                    last_pump_time = now

            last_loss = epoch_loss / max(1, num_samples)
            last_acc = epoch_correct / max(1, num_samples)

            if epoch_callback is not None:
                epoch_callback(epoch + 1, config.epochs, float(last_loss), float(last_acc))
                last_pump_time = time.time()

        policy.eval()
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": float(last_loss),
            "accuracy": float(last_acc),
            "steps": num_samples,
            "episodes": demo_buffer.total_episodes,
        }


class DQNImitationLearner(BaseImitationLearner):
    """Behavioral cloning strategy for Stable-Baselines3 DQN.

    Trains the Q-network via cross-entropy classification over Q-values
    to favor actions demonstrated by the human expert.
    """

    def train(
        self,
        model: Any,
        demo_buffer: DemonstrationBuffer,
        config: ImitationConfig,
        epoch_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Dict[str, float]:
        if demo_buffer.is_empty():
            return {"loss": 0.0, "accuracy": 0.0, "steps": 0}

        all_obs, all_actions = demo_buffer.get_flat_dataset()
        device = model.device
        q_net = model.q_net
        q_net.train()

        optimizer = torch.optim.AdamW(
            q_net.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        num_samples = len(all_actions)
        last_loss = 0.0
        last_acc = 0.0
        last_pump_time = time.time()

        for epoch in range(config.epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_correct = 0

            for start_idx in range(0, num_samples, config.batch_size):
                batch_idx = indices[start_idx : start_idx + config.batch_size]
                obs_b = torch.as_tensor(all_obs[batch_idx], dtype=torch.float32, device=device)
                actions_b = torch.as_tensor(all_actions[batch_idx], dtype=torch.int64, device=device)

                optimizer.zero_grad()
                q_values = q_net(obs_b)
                loss = F.cross_entropy(q_values, actions_b)
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                optimizer.step()

                preds = q_values.argmax(dim=-1)
                epoch_correct += (preds == actions_b).sum().item()
                epoch_loss += loss.item() * len(batch_idx)

                now = time.time()
                if now - last_pump_time > 0.5:
                    try:
                        import pygame
                        pygame.event.pump()
                    except Exception:
                        pass
                    last_pump_time = now

            last_loss = epoch_loss / max(1, num_samples)
            last_acc = epoch_correct / max(1, num_samples)

            if epoch_callback is not None:
                epoch_callback(epoch + 1, config.epochs, float(last_loss), float(last_acc))
                last_pump_time = time.time()

        # Synchronize target Q-network
        if hasattr(model, "q_net_target") and model.q_net_target is not None:
            model.q_net_target.load_state_dict(model.q_net.state_dict())

        q_net.eval()
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "loss": float(last_loss),
            "accuracy": float(last_acc),
            "steps": num_samples,
            "episodes": demo_buffer.total_episodes,
        }



class ImitationLearnerFactory:
    """Factory responsible for instantiating the appropriate imitation learner strategy."""

    _learners: Dict[Any, BaseImitationLearner] = {}

    @classmethod
    def register(cls, algo_type: Any, learner: BaseImitationLearner) -> None:
        cls._learners[algo_type] = learner
        if hasattr(algo_type, "value"):
            cls._learners[algo_type.value] = learner
        elif isinstance(algo_type, str):
            cls._learners[algo_type.upper()] = learner

    @classmethod
    def create(cls, algorithm: Union[AlgorithmType, str]) -> BaseImitationLearner:
        learner = cls._learners.get(algorithm)
        if not learner and hasattr(algorithm, "value"):
            learner = cls._learners.get(algorithm.value)
        if not learner and isinstance(algorithm, str):
            learner = cls._learners.get(algorithm.upper())

        if not learner:
            supported = list(cls._learners.keys())
            raise ValueError(
                f"Unsupported algorithm '{algorithm}' for imitation learning. Registered: {supported}"
            )
        return learner


# Auto-register core imitation learner strategies
ImitationLearnerFactory.register(AlgorithmType.GRPO, GRPOImitationLearner())
ImitationLearnerFactory.register(AlgorithmType.PPO, PPOImitationLearner())
ImitationLearnerFactory.register(AlgorithmType.DQN, DQNImitationLearner())
