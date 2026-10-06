"""Unit tests for Human Demonstration Collection & Behavioral Cloning (BC) Pre-training.

Verifies:
1. DemonstrationBuffer thread-safe recording, trajectory packaging, and flattening.
2. Behavioral Cloning training on GRPOActor (Causal Transformer sequence mode and step mode).
3. Behavioral Cloning on PPO and DQN architectures.
4. Key event mapping, human demonstration step execution, and Space key transition to RL.
5. Zero-demo graceful fallback.
"""
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


import numpy as np
import pygame
import torch
import torch.nn as nn

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl_game.engine.game import GameEngine, GamePhase
from rl_game.engine.registry import SobjectRegistry
from rl_game.engine.sobject import Sobject
from rl_game.rl import (
    AlgorithmType,
    DemonstrationBuffer,
    DQNConfig,
    GRPOConfig,
    GridRLEnv,
    ImitationConfig,
    ImitationLearnerFactory,
    PPOConfig,
    RLConfig,
    RLModelFactory,
    TemporalConfig,
    TemporalType,
)


class DummyPlayer(Sobject):
    def handle_AI_input(self, action: int, field) -> None:
        pass


class MockGame(GameEngine):
    def __init__(self, width: int = 10, height: int = 10, cell_size: int = 20):
        super().__init__(width, height, cell_size, mode="ai")

    def register_objects(self, registry: SobjectRegistry) -> None:
        registry.register(DummyPlayer, 1, (0, 255, 0))
        registry.register(Sobject, 2, (0, 0, 255))
        registry.register(Sobject, 3, (255, 0, 0))

    def setup_field(self) -> None:
        player = self.registry.create(1, 2, 2, self.cell_size)
        self.set_player(player)

    def update(self) -> None:
        pass


class MockEnv(GridRLEnv):
    def reward(self):
        return 1.0, False, False

    def goal(self):
        pass


class TestDemonstrationBuffer(unittest.TestCase):
    """Tests DemonstrationBuffer storage, sequencing, and batching."""

    def test_buffer_recording_and_finish_episode(self):
        buf = DemonstrationBuffer()
        self.assertTrue(buf.is_empty())
        self.assertEqual(buf.total_steps, 0)
        self.assertEqual(buf.total_episodes, 0)

        # Episode 1: 3 steps, terminated
        obs_shape = (4, 10, 10)
        buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=0.0, terminated=False)
        buf.add_step(np.ones(obs_shape, dtype=np.uint8), action=1, reward=10.0, terminated=False)
        buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=2, reward=20.0, terminated=True)

        self.assertEqual(buf.total_steps, 3)
        self.assertEqual(buf.total_episodes, 1)

        # Episode 2: 2 steps, truncated
        buf.add_step(np.ones(obs_shape, dtype=np.uint8), action=3, reward=-1.0, terminated=False)
        buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=-1.0, truncated=True)

        self.assertEqual(buf.total_steps, 5)
        self.assertEqual(buf.total_episodes, 2)

        # Trajectories extraction
        trajectories = buf.get_trajectories()
        self.assertEqual(len(trajectories), 2)
        self.assertEqual(trajectories[0][0].shape, (3, 4, 10, 10))
        self.assertTrue(np.array_equal(trajectories[0][1], np.array([0, 1, 2])))
        self.assertEqual(trajectories[1][0].shape, (2, 4, 10, 10))
        self.assertTrue(np.array_equal(trajectories[1][1], np.array([3, 0])))

        # Flat dataset extraction
        all_obs, all_actions = buf.get_flat_dataset()
        self.assertEqual(all_obs.shape, (5, 4, 10, 10))
        self.assertEqual(all_actions.shape, (5,))

        # Clear buffer
        buf.clear()
        self.assertTrue(buf.is_empty())
        self.assertEqual(buf.total_steps, 0)
        self.assertEqual(buf.total_episodes, 0)

    def test_multithreaded_recording_safety(self):
        """Verifies DemonstrationBuffer thread safety under concurrent writes."""
        import threading
        buf = DemonstrationBuffer()
        obs_shape = (4, 5, 5)

        def worker(w_id: int):
            for i in range(20):
                buf.add_step(
                    np.zeros(obs_shape, dtype=np.uint8),
                    action=w_id % 4,
                    reward=1.0,
                    terminated=(i == 19),
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(buf.total_steps, 100)
        self.assertEqual(buf.total_episodes, 5)

    def test_demonstration_persistence_save_and_load(self):
        """Verifies that episodes auto-save to disk and load accurately into a new buffer instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf1 = DemonstrationBuffer(save_dir=tmpdir)
            obs_shape = (4, 8, 8)

            # Record 2 episodes
            for step in range(3):
                buf1.add_step(np.full(obs_shape, step, dtype=np.uint8), action=step % 4, reward=1.0, terminated=(step == 2))
            for step in range(4):
                buf1.add_step(np.full(obs_shape, step + 10, dtype=np.uint8), action=(step + 1) % 4, reward=2.0, terminated=(step == 3))

            self.assertEqual(buf1.total_steps, 7)
            self.assertEqual(buf1.total_episodes, 2)

            # Verify files exist on disk
            saved_files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
            self.assertEqual(len(saved_files), 2)

            # New session: create new buffer pointing to the same directory
            buf2 = DemonstrationBuffer(save_dir=tmpdir)
            self.assertEqual(buf2.total_steps, 7)
            self.assertEqual(buf2.total_episodes, 2)

            # Check data equality
            traj1 = buf1.get_trajectories()
            traj2 = buf2.get_trajectories()
            for (obs1, act1), (obs2, act2) in zip(traj1, traj2):
                self.assertTrue(np.array_equal(obs1, obs2))
                self.assertTrue(np.array_equal(act1, act2))

    def test_cross_session_incremental_merge(self):
        """Verifies that new demonstrations in a new session merge seamlessly with past demonstrations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: 5 steps
            buf1 = DemonstrationBuffer(save_dir=tmpdir)
            obs_shape = (4, 8, 8)
            for step in range(5):
                buf1.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=1.0, terminated=(step == 4))

            # Session 2: loads 5 steps, then records another 3 steps
            buf2 = DemonstrationBuffer(save_dir=tmpdir)
            self.assertEqual(buf2.total_steps, 5)

            for step in range(3):
                buf2.add_step(np.ones(obs_shape, dtype=np.uint8), action=1, reward=2.0, terminated=(step == 2))

            self.assertEqual(buf2.total_steps, 8)
            self.assertEqual(buf2.total_episodes, 2)

            # Session 3: loads all 8 steps (both sessions merged)
            buf3 = DemonstrationBuffer(save_dir=tmpdir)
            self.assertEqual(buf3.total_steps, 8)
            self.assertEqual(buf3.total_episodes, 2)

    def test_corrupted_file_resilience(self):
        """Verifies that corrupted files in save_dir are skipped safely without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid file with 2 steps (>= min_episode_steps)
            buf = DemonstrationBuffer(save_dir=tmpdir, min_episode_steps=2)
            obs_shape = (4, 5, 5)
            buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=1, reward=1.0, terminated=False)
            buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=1, reward=1.0, terminated=True)

            # Create a corrupted file
            corrupt_path = os.path.join(tmpdir, "demo_corrupted.npz")
            with open(corrupt_path, "wb") as f:
                f.write(b"NOT_A_VALID_ZIP_OR_NPZ_DATA")

            # Load into new buffer
            buf_resilient = DemonstrationBuffer(save_dir=tmpdir, min_episode_steps=2)
            # Should have loaded the 1 valid episode without crashing
            self.assertEqual(buf_resilient.total_episodes, 1)
            self.assertEqual(buf_resilient.total_steps, 2)

    def test_subminimal_fragment_rejection(self):
        """Verifies that single-step mispresses or sub-minimal fragments are rejected and not saved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = DemonstrationBuffer(save_dir=tmpdir, min_episode_steps=2)
            obs_shape = (4, 5, 5)
            # Add 1 step and terminate
            buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=0.0, terminated=True)

            self.assertEqual(buf.total_episodes, 0)
            self.assertEqual(len(os.listdir(tmpdir)), 0)

            # Now add 2 steps and finish
            buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=0.0, terminated=False)
            buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=1, reward=1.0, terminated=True)

            self.assertEqual(buf.total_episodes, 1)
            self.assertEqual(len([f for f in os.listdir(tmpdir) if f.endswith(".npz")]), 1)

    def test_max_episodes_bounding(self):
        """Verifies that only the most recent max_episodes are loaded into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            obs_shape = (4, 5, 5)
            # Save 5 episodes directly
            for i in range(5):
                buf = DemonstrationBuffer(save_dir=tmpdir, min_episode_steps=2)
                buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=0, reward=0.0, terminated=False)
                buf.add_step(np.zeros(obs_shape, dtype=np.uint8), action=1, reward=1.0, terminated=True)

            self.assertEqual(len([f for f in os.listdir(tmpdir) if f.startswith("demo_")]), 5)

            # Load with max_episodes=2
            bounded_buf = DemonstrationBuffer(save_dir=tmpdir, max_episodes=2, min_episode_steps=2)
            self.assertEqual(bounded_buf.total_episodes, 2)
            self.assertEqual(bounded_buf.total_steps, 4)

    def test_orphaned_tmp_files_cleanup(self):
        """Verifies that orphan tmp_*.npz files from past process kills are cleaned up on startup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orphan_path = os.path.join(tmpdir, "tmp_deadbeef_demo_test.npz")
            with open(orphan_path, "wb") as f:
                f.write(b"orphaned data")

            self.assertTrue(os.path.exists(orphan_path))
            buf = DemonstrationBuffer(save_dir=tmpdir)
            self.assertFalse(os.path.exists(orphan_path))

    def test_shape_and_action_mismatch_resilience(self):
        """Verifies that files with mismatched observation shape or invalid actions are safely skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Valid file: shape (2, 4, 5, 5), actions [0, 1]
            valid_path = os.path.join(tmpdir, "demo_20260101_00000000000000000001_aaa_2steps.npz")
            np.savez_compressed(
                valid_path,
                obs=np.zeros((2, 4, 5, 5), dtype=np.uint8),
                actions=np.array([0, 1], dtype=np.int64),
                rewards=np.array([0.0, 1.0], dtype=np.float32),
            )

            # 2. Shape mismatch: shape (2, 8, 5, 5) (channel mismatch)
            mismatch_path = os.path.join(tmpdir, "demo_20260101_00000000000000000002_bbb_2steps.npz")
            np.savez_compressed(
                mismatch_path,
                obs=np.zeros((2, 8, 5, 5), dtype=np.uint8),
                actions=np.array([0, 1], dtype=np.int64),
                rewards=np.array([0.0, 1.0], dtype=np.float32),
            )

            # 3. Action mismatch: negative action [-1, 99]
            invalid_act_path = os.path.join(tmpdir, "demo_20260101_00000000000000000003_ccc_2steps.npz")
            np.savez_compressed(
                invalid_act_path,
                obs=np.zeros((2, 4, 5, 5), dtype=np.uint8),
                actions=np.array([-1, 99], dtype=np.int64),
                rewards=np.array([0.0, 1.0], dtype=np.float32),
            )

            buf = DemonstrationBuffer(save_dir=tmpdir, min_episode_steps=2)
            # Only the valid file should be loaded
            self.assertEqual(buf.total_episodes, 1)
            self.assertEqual(buf.total_steps, 2)




class TestImitationLearningBehavioralCloning(unittest.TestCase):
    """Tests behavioral cloning pre-training across model architectures."""

    def setUp(self):
        self.game = MockGame(width=10, height=10, cell_size=20)

    def test_grpo_transformer_behavioral_cloning(self):
        """Verifies BC optimization on GRPO Causal Transformer policy."""
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            temporal=TemporalConfig(
                enabled=True,
                full_episode=True,
                max_seq_len=20,
                temporal_type=TemporalType.TRANSFORMER,
                hidden_dim=32,
                features_dim=32,
                n_heads=2,
                n_layers=1,
                dim_feedforward=64,
            ),
            imitation=ImitationConfig(
                enabled=True,
                epochs=20,
                batch_size=16,
                learning_rate=3e-3,
            ),
        )
        env = MockEnv(game=self.game, config=config)
        model = RLModelFactory.create(config=config, env=env)

        # Create expert demonstrations consistently choosing action 2 (LEFT)
        buf = DemonstrationBuffer()
        for ep in range(3):
            obs, _ = env.reset()
            for step in range(5):
                next_obs, reward, term, trunc, _ = env.step(2)
                buf.add_step(obs, action=2, reward=reward, terminated=(step == 4))
                obs = next_obs

        learner = ImitationLearnerFactory.create(AlgorithmType.GRPO)
        metrics = learner.train(model, buf, config.imitation)

        self.assertIn("loss", metrics)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0.70)  # Should achieve high accuracy on expert action
        self.assertEqual(metrics["steps"], 15)

        # Verify reference actor weights were synchronized with active actor
        for p1, p2 in zip(model.actor.parameters(), model.ref_actor.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Verify inference predicts action 2
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertEqual(int(action), 2)

    def test_ppo_behavioral_cloning(self):
        """Verifies BC optimization on SB3 PPO policy."""
        config = RLConfig(
            algorithm=AlgorithmType.PPO,
            action_space=4,
            max_id=4,
            temporal=TemporalConfig(
                enabled=True,
                temporal_type=TemporalType.STACK,
                spatial_channels=(16, 32),
                features_dim=32,
            ),
            ppo=PPOConfig(n_steps=64, batch_size=32),
            imitation=ImitationConfig(epochs=15, batch_size=16, learning_rate=5e-3),
        )
        env = MockEnv(game=self.game, config=config)
        model = RLModelFactory.create(config=config, env=env)

        obs, _ = env.reset()
        buf = DemonstrationBuffer()
        for step in range(20):
            next_obs, reward, terminated, truncated, _ = env.step(1)
            buf.add_step(obs, action=1, reward=reward, terminated=(step % 5 == 4))
            obs = next_obs
            if step % 5 == 4:
                obs, _ = env.reset()

        learner = ImitationLearnerFactory.create(AlgorithmType.PPO)
        metrics = learner.train(model, buf, config.imitation)

        self.assertGreater(metrics["accuracy"], 0.70)
        self.assertEqual(metrics["steps"], 20)

    def test_dqn_behavioral_cloning(self):
        """Verifies BC cross-entropy classification on SB3 DQN Q-network."""
        config = RLConfig(
            algorithm=AlgorithmType.DQN,
            action_space=4,
            max_id=4,
            temporal=TemporalConfig(
                enabled=True,
                temporal_type=TemporalType.STACK,
                spatial_channels=(16, 32),
                features_dim=32,
            ),
            dqn=DQNConfig(buffer_size=1000, learning_starts=10),
            imitation=ImitationConfig(epochs=15, batch_size=16, learning_rate=5e-3),
        )
        env = MockEnv(game=self.game, config=config)
        model = RLModelFactory.create(config=config, env=env)

        obs, _ = env.reset()
        buf = DemonstrationBuffer()
        for step in range(20):
            next_obs, reward, terminated, truncated, _ = env.step(3)
            buf.add_step(obs, action=3, reward=reward, terminated=(step % 5 == 4))
            obs = next_obs
            if step % 5 == 4:
                obs, _ = env.reset()

        learner = ImitationLearnerFactory.create(AlgorithmType.DQN)
        metrics = learner.train(model, buf, config.imitation)

        self.assertGreater(metrics["accuracy"], 0.70)
        self.assertEqual(metrics["steps"], 20)



class TestGameEngineDemonstrationAndTransitionFlow(unittest.TestCase):
    """Tests the interaction loop: manual arrow steps -> demonstration buffer -> space key -> RL."""

    def setUp(self):
        self.game = MockGame(width=10, height=10, cell_size=20)
        self.config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            temporal=TemporalConfig(
                enabled=True,
                full_episode=True,
                max_seq_len=20,
                temporal_type=TemporalType.TRANSFORMER,
                hidden_dim=32,
                features_dim=32,
                n_heads=2,
                n_layers=1,
            ),
            imitation=ImitationConfig(enabled=True, epochs=2, min_demos=1),
        )
        self.env = MockEnv(game=self.game, config=self.config)
        self.model = RLModelFactory.create(config=self.config, env=self.env)

    def test_key_to_action_mapping(self):
        self.assertEqual(self.game._key_to_action(pygame.K_UP), 0)
        self.assertEqual(self.game._key_to_action(pygame.K_DOWN), 1)
        self.assertEqual(self.game._key_to_action(pygame.K_LEFT), 2)
        self.assertEqual(self.game._key_to_action(pygame.K_RIGHT), 3)
        self.assertIsNone(self.game._key_to_action(pygame.K_a))

    def test_human_demo_step_and_buffer_population(self):
        self.game.phase = GamePhase.HUMAN_DEMO
        self.game.demo_buffer = DemonstrationBuffer()
        self.game.env = self.env
        self.game.current_obs, _ = self.env.reset()

        # Simulate 3 arrow key steps
        self.game._step_human_demo(0)  # UP
        self.game._step_human_demo(2)  # LEFT
        self.game._step_human_demo(3)  # RIGHT

        self.assertEqual(self.game.demo_buffer.total_steps, 3)
        self.assertFalse(self.game.demo_buffer.is_empty())

    def test_space_key_transition_and_rl_worker_launch(self):
        """Verifies real daemon worker thread launch and clean stop without mocking."""
        self.game.phase = GamePhase.HUMAN_DEMO
        self.game.demo_buffer = DemonstrationBuffer()
        self.game.env = self.env
        self.game.config = self.config
        self.game.model = self.model
        self.game.total_timesteps = 10
        self.game.current_obs, _ = self.env.reset()

        # Add 2 demo steps
        self.game._step_human_demo(1)
        self.game._step_human_demo(1)

        # Trigger SPACE key transition with real ai_work thread
        self.game._transition_to_pretraining_and_rl()

        self.assertEqual(self.game.phase, GamePhase.RL_TRAINING)
        self.assertIsNotNone(self.game.worker_thread)
        self.assertTrue(self.game.worker_thread.daemon)  # Must be daemon to prevent zombie process

        # Signal stop immediately and verify worker terminates cleanly within 2 seconds
        self.game.stop_event.set()
        self.game.worker_thread.join(timeout=2.0)
        self.assertFalse(self.game.worker_thread.is_alive())

    def test_zero_demo_graceful_fallback(self):
        """Verifies that pressing Space with 0 demo steps gracefully transitions to RL without error."""
        self.game.phase = GamePhase.HUMAN_DEMO
        self.game.demo_buffer = DemonstrationBuffer()  # 0 steps
        self.game.env = self.env
        self.game.config = self.config
        self.game.model = self.model
        self.game.total_timesteps = 10
        self.game.current_obs, _ = self.env.reset()

        self.game._transition_to_pretraining_and_rl()
        self.assertEqual(self.game.phase, GamePhase.RL_TRAINING)
        self.game.stop_event.set()
        self.game.worker_thread.join(timeout=2.0)
        self.assertFalse(self.game.worker_thread.is_alive())

    def test_low_quality_demo_ref_actor_protection(self):
        """Verifies that ref_actor is NOT overwritten when BC accuracy < 50% (policy degeneration guard)."""
        learner = ImitationLearnerFactory.create(AlgorithmType.GRPO)
        # Create initial reference actor weights
        initial_ref_weights = [p.clone() for p in self.model.ref_actor.parameters()]

        # Low-quality demo: 2 steps with random noise, 0 epochs or mislabeled
        buf = DemonstrationBuffer(min_episode_steps=2)
        obs, _ = self.env.reset()
        buf.add_step(obs, action=0, reward=-50.0, terminated=False)
        buf.add_step(obs, action=0, reward=-50.0, terminated=True)

        low_config = ImitationConfig(enabled=True, epochs=1, min_demos=1, learning_rate=1e-5)
        # Force low accuracy by evaluating noisy steps
        metrics = learner.train(self.model, buf, low_config)

        # If accuracy < 50%, ref_actor weights must remain identical to initial
        if metrics["accuracy"] < 0.5:
            for p_init, p_current in zip(initial_ref_weights, self.model.ref_actor.parameters()):
                self.assertTrue(torch.allclose(p_init, p_current))

    def test_epoch_callback_invoked(self):
        """Verifies that epoch_callback is invoked on each epoch for OS event pumping."""
        learner = ImitationLearnerFactory.create(AlgorithmType.GRPO)
        buf = DemonstrationBuffer(min_episode_steps=2)
        obs, _ = self.env.reset()
        buf.add_step(obs, action=1, reward=1.0, terminated=False)
        buf.add_step(obs, action=1, reward=1.0, terminated=True)

        callbacks_received = []
        def mock_callback(ep, total_ep, loss, acc):
            callbacks_received.append((ep, total_ep, loss, acc))

        config = ImitationConfig(enabled=True, epochs=3, min_demos=1)
        learner.train(self.model, buf, config, epoch_callback=mock_callback)

        self.assertEqual(len(callbacks_received), 3)
        self.assertEqual(callbacks_received[-1][0], 3)
        self.assertEqual(callbacks_received[-1][1], 3)

    def test_transformer_sequence_truncation(self):
        """Verifies that trajectories exceeding max_seq_len are safely truncated without crashing."""
        learner = ImitationLearnerFactory.create(AlgorithmType.GRPO)
        buf = DemonstrationBuffer(min_episode_steps=2)
        obs, _ = self.env.reset()
        # Feed 30 steps into a model with max_seq_len=20
        for step in range(30):
            buf.add_step(obs, action=1, reward=1.0, terminated=(step == 29))

        config = ImitationConfig(enabled=True, epochs=1, min_demos=1)
        metrics = learner.train(self.model, buf, config)
        self.assertIn("accuracy", metrics)
        self.assertIn("loss", metrics)

    def test_weight_decay_validation(self):
        with self.assertRaises(ValueError):
            ImitationConfig(weight_decay=-0.01)


if __name__ == "__main__":
    unittest.main()

