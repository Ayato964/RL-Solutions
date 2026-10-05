"""Comprehensive test suite for RL temporal system, PPO, DQN, and environment integration.
"""
import os
import unittest
import numpy as np
import torch
import pygame

# Initialize pygame headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

from rl_game.rl import (
    RLConfig,
    TemporalConfig,
    AlgorithmType,
    TemporalType,
    PPOConfig,
    DQNConfig,
    GRPOConfig,
    RLModelFactory,
    GridRLEnv,
    TemporalObservationBuffer,
    TemporalLSTMFeaturesExtractor,
    TemporalGRUFeaturesExtractor,
    TemporalConv1dFeaturesExtractor,
    TemporalStackedFeaturesExtractor,
    TemporalTransformerFeaturesExtractor,
    AIPlayer,
    GRPO,
    get_features_extractor_specs,
)
from rl_game.rl.DQN import DQNTrainer as LegacyDQNTrainer, AIPlayer as LegacyAIPlayer
from snake_game.main import BadAppleGame, BadAppleEnv, BadAppleTrainer, create_default_config


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = create_default_config()
        self.assertEqual(cfg.algorithm, AlgorithmType.GRPO)
        self.assertTrue(cfg.temporal.enabled)
        self.assertTrue(cfg.temporal.full_episode)
        self.assertEqual(cfg.temporal.n_frames, 1)
        self.assertEqual(cfg.temporal.max_seq_len, 1000)
        self.assertEqual(cfg.temporal.temporal_type, TemporalType.TRANSFORMER)
        self.assertEqual(cfg.temporal.n_heads, 4)
        self.assertEqual(cfg.temporal.n_layers, 2)
        self.assertEqual(cfg.temporal.spatial_channels, (64, 128))
        self.assertEqual(cfg.temporal.features_dim, 128)
        self.assertEqual(cfg.grpo.group_size, 32)
        self.assertEqual(cfg.grpo.n_epochs, 1)
        self.assertIsNone(cfg.grpo.gradient_accumulation_steps)
        self.assertEqual(cfg.grpo.advantage_mode, "step_discounted")
        self.assertIsNone(cfg.grpo.rollout_steps)

    def test_switch_algorithm(self):
        cfg = create_default_config()
        cfg.algorithm = "DQN"
        self.assertEqual(cfg.algorithm, AlgorithmType.DQN)
        cfg.algorithm = AlgorithmType.PPO
        self.assertEqual(cfg.algorithm, AlgorithmType.PPO)
        cfg.algorithm = AlgorithmType.GRPO
        self.assertEqual(cfg.algorithm, AlgorithmType.GRPO)

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            TemporalConfig(n_frames=0)
        with self.assertRaises(ValueError):
            RLConfig(action_space=0)
        with self.assertRaises(ValueError):
            TemporalConfig(hidden_dim=65, n_heads=4)  # Not divisible
        with self.assertRaises(ValueError):
            TemporalConfig(aggregation="invalid_agg")
        with self.assertRaises(ValueError):
            GRPOConfig(group_size=0)
        with self.assertRaises(ValueError):
            GRPOConfig(n_epochs=0)
        with self.assertRaises(ValueError):
            GRPOConfig(gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            GRPOConfig(advantage_mode="invalid_mode")


class TestTemporalBuffer(unittest.TestCase):
    def test_buffer_sliding_window(self):
        buf = TemporalObservationBuffer(n_frames=3, enabled=True)
        f1 = np.ones((2, 4, 4), dtype=np.uint8) * 1
        f2 = np.ones((2, 4, 4), dtype=np.uint8) * 2
        f3 = np.ones((2, 4, 4), dtype=np.uint8) * 3
        f4 = np.ones((2, 4, 4), dtype=np.uint8) * 4

        # Reset fills buffer
        obs = buf.reset(f1)
        self.assertEqual(obs.shape, (6, 4, 4))
        self.assertTrue(np.all(obs[:2] == 1))
        self.assertTrue(np.all(obs[2:4] == 1))
        self.assertTrue(np.all(obs[4:6] == 1))

        # Append f2
        obs = buf.append(f2)
        self.assertTrue(np.all(obs[:2] == 1))
        self.assertTrue(np.all(obs[2:4] == 1))
        self.assertTrue(np.all(obs[4:6] == 2))

        # Append f3
        obs = buf.append(f3)
        self.assertTrue(np.all(obs[:2] == 1))
        self.assertTrue(np.all(obs[2:4] == 2))
        self.assertTrue(np.all(obs[4:6] == 3))

        # Append f4 (f1 is completely pushed out)
        obs = buf.append(f4)
        self.assertTrue(np.all(obs[:2] == 2))
        self.assertTrue(np.all(obs[2:4] == 3))
        self.assertTrue(np.all(obs[4:6] == 4))

    def test_buffer_disabled(self):
        buf = TemporalObservationBuffer(n_frames=4, enabled=False)
        f1 = np.ones((2, 4, 4), dtype=np.uint8)
        obs = buf.reset(f1)
        self.assertEqual(obs.shape, (2, 4, 4))


class TestFeatureExtractors(unittest.TestCase):
    def setUp(self):
        from gymnasium import spaces
        self.n_frames = 4
        self.raw_channels = 4
        self.h, self.w = 10, 15
        self.obs_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_frames * self.raw_channels, self.h, self.w),
            dtype=np.uint8,
        )
        self.dummy_input = torch.randint(
            0, 2,
            size=(2, self.n_frames * self.raw_channels, self.h, self.w),
            dtype=torch.uint8,
        )

    def test_lstm_extractor(self):
        ext = TemporalLSTMFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            features_dim=16,
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        # Verify gradient flow
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(ext.lstm.weight_ih_l0.grad)

    def test_gru_extractor(self):
        ext = TemporalGRUFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            features_dim=16,
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(ext.gru.weight_ih_l0.grad)

    def test_conv1d_extractor(self):
        ext = TemporalConv1dFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            features_dim=16,
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()

    def test_stacked_extractor(self):
        ext = TemporalStackedFeaturesExtractor(
            self.obs_space,
            features_dim=16,
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()

    def test_transformer_extractor_last_head(self):
        ext = TemporalTransformerFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            n_heads=4,
            n_layers=2,
            features_dim=16,
            aggregation="last",
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(ext.transformer.layers[0].self_attn.in_proj_weight.grad)
        self.assertIsNotNone(ext.pos_embedding.grad)

    def test_transformer_extractor_mean_head(self):
        ext = TemporalTransformerFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            n_heads=4,
            n_layers=2,
            features_dim=16,
            aggregation="mean",
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()

    def test_transformer_extractor_attention_head(self):
        ext = TemporalTransformerFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=32,
            n_heads=4,
            n_layers=2,
            features_dim=16,
            aggregation="attention",
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 16))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(ext.attn_query.grad)


class TestRLTrainingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    def test_environment_and_ppo_training(self):
        config = RLConfig(
            algorithm=AlgorithmType.PPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            temporal=TemporalConfig(
                enabled=True,
                n_frames=4,
                temporal_type=TemporalType.LSTM,
                hidden_dim=32,
                features_dim=16,
            ),
            ppo=PPOConfig(
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                verbose=0,
                tensorboard_log=None,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)

        # Test reset and step
        obs, info = env.reset()
        self.assertEqual(obs.shape, (16, 10, 10))
        next_obs, reward, term, trunc, _ = env.step(1)
        self.assertEqual(next_obs.shape, (16, 10, 10))
        self.assertIsInstance(reward, float)

        # Test PPO creation and fast learn
        model = RLModelFactory.create(config, env)
        model.learn(total_timesteps=32)
        self.assertTrue(True)

    def test_environment_and_transformer_ppo_training(self):
        config = RLConfig(
            algorithm=AlgorithmType.PPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            temporal=TemporalConfig(
                enabled=True,
                n_frames=4,
                temporal_type=TemporalType.TRANSFORMER,
                hidden_dim=32,
                n_heads=4,
                n_layers=2,
                features_dim=16,
                aggregation="last",
            ),
            ppo=PPOConfig(
                n_steps=32,
                batch_size=16,
                n_epochs=2,
                verbose=0,
                tensorboard_log=None,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)

        model = RLModelFactory.create(config, env)
        model.learn(total_timesteps=32)
        self.assertTrue(True)

    def test_environment_and_dqn_training(self):
        config = RLConfig(
            algorithm=AlgorithmType.DQN,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            temporal=TemporalConfig(
                enabled=True,
                n_frames=4,
                temporal_type=TemporalType.GRU,
                hidden_dim=32,
                features_dim=16,
            ),
            dqn=DQNConfig(
                buffer_size=1000,
                learning_starts=10,
                batch_size=16,
                train_freq=4,
                verbose=0,
                tensorboard_log=None,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)

        model = RLModelFactory.create(config, env)
        model.learn(total_timesteps=16)
        self.assertTrue(True)

    def test_environment_and_transformer_grpo_training(self):
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            temporal=TemporalConfig(
                enabled=True,
                n_frames=4,
                temporal_type=TemporalType.TRANSFORMER,
                hidden_dim=32,
                n_heads=4,
                n_layers=2,
                features_dim=16,
                aggregation="last",
            ),
            grpo=GRPOConfig(
                group_size=2,
                rollout_steps=8,
                batch_size=8,
                n_epochs=1,
                verbose=0,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)

        model = RLModelFactory.create(config, env)
        # Verify Critic-free architecture
        self.assertFalse(hasattr(model, "critic"))
        self.assertFalse(hasattr(model, "value_net"))
        self.assertTrue(hasattr(model, "actor"))

        # Test learning and prediction
        model.learn(total_timesteps=16)
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        self.assertIn(action, [0, 1, 2, 3])

        # Test save and load
        model.save("model/test_grpo_ckpt.pt")
        loaded_model = GRPO.load("model/test_grpo_ckpt.pt", env=env, config=config)
        action_loaded, _ = loaded_model.predict(obs, deterministic=True)
        self.assertEqual(action, action_loaded)

    def test_grpo_early_termination_with_callback(self):
        """Verifies GRPO.learn() terminates immediately when callback signals stop."""
        import threading
        from rl_game.engine.game import GracefulStopCallback

        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            grpo=GRPOConfig(
                group_size=4,
                rollout_steps=100,
                verbose=0,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)
        model = RLModelFactory.create(config, env)

        stop_event = threading.Event()
        callback = GracefulStopCallback(stop_event)

        # Trigger stop immediately
        stop_event.set()
        # Request 1,000,000 steps; must exit immediately and NOT collect 1M steps
        model.learn(total_timesteps=1_000_000, callback=callback)
        self.assertLess(model.num_timesteps, 50)

    def test_grpo_analytical_kl_divergence(self):
        """Verifies exact analytical categorical KL divergence is non-negative and zero for identical policies."""
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            grpo=GRPOConfig(kl_coef=0.01),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)
        model = RLModelFactory.create(config, env)

        dummy_obs = torch.zeros((4, config.max_id * config.temporal.n_frames, 10, 10))
        dist = model.actor(dummy_obs)
        ref_dist = model.ref_actor(dummy_obs)

        # Initialized identically, KL should be 0.0
        raw_kl = torch.distributions.kl.kl_divergence(dist, ref_dist)
        kl = torch.clamp(raw_kl, min=0.0).mean()
        self.assertAlmostEqual(kl.item(), 0.0, places=5)
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_backward_compatibility(self):
        class ConcreteLegacyTrainer(LegacyDQNTrainer):
            def reward(self):
                return 0.0, False, False

            def goal(self):
                pass

        game = BadAppleGame(width=10, height=10, cell_size=20)
        legacy_trainer = ConcreteLegacyTrainer(game=game, action_space=4, max_id=4)
        obs, info = legacy_trainer.reset()
        self.assertEqual(obs.shape, (4, 10, 10))
        self.assertTrue(issubclass(BadAppleTrainer, GridRLEnv))

        # Test legacy positional instantiation of BadAppleTrainer
        legacy_bad_apple = BadAppleTrainer(game, 4, 4)
        obs, _ = legacy_bad_apple.reset()
        self.assertEqual(obs.shape, (4, 10, 10))


class TestAdversarialVerificationFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()

    def test_normalize_images_is_false(self):
        """Verifies SB3 does not divide categorical one-hot inputs by 255.0."""
        config = create_default_config()
        config.algorithm = AlgorithmType.PPO
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)

        ppo_model = RLModelFactory.create(config, env)
        self.assertFalse(ppo_model.policy_kwargs.get("normalize_images", True))

        config.algorithm = AlgorithmType.DQN
        dqn_model = RLModelFactory.create(config, env)
        self.assertFalse(dqn_model.policy_kwargs.get("normalize_images", True))

    def test_legacy_ai_player_sb3_compatibility(self):
        """Verifies AIPlayer without temporal mode initializes SB3 optimizers without crashing."""
        config = create_default_config()
        config.algorithm = AlgorithmType.PPO
        config.temporal.enabled = False
        config.ppo.n_steps = 16
        config.ppo.batch_size = 16
        config.ppo.n_epochs = 1
        config.ppo.verbose = 0
        config.ppo.tensorboard_log = None

        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)
        model = RLModelFactory.create(config, env)
        # Should learn without ValueError: can't optimize uninitialized parameter
        model.learn(total_timesteps=16)
        self.assertTrue(True)

    def test_determinism_and_seeding(self):
        """Verifies Gymnasium seed reproducibility contract."""
        config = create_default_config()
        game1 = BadAppleGame(width=15, height=15, cell_size=20)
        env1 = BadAppleEnv(game=game1, config=config)
        obs1, _ = env1.reset(seed=42)

        game2 = BadAppleGame(width=15, height=15, cell_size=20)
        env2 = BadAppleEnv(game=game2, config=config)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2)

        game3 = BadAppleGame(width=15, height=15, cell_size=20)
        env3 = BadAppleEnv(game=game3, config=config)
        obs3, _ = env3.reset(seed=999)
        self.assertFalse(np.array_equal(obs1, obs3))

    def test_bounds_check_and_exploit_prevention(self):
        """Verifies negative indices do not trigger ghost pickups or crashes."""
        game = BadAppleGame(width=10, height=10, cell_size=20)
        # Set player at boundary (0, 0)
        game.player.x = 0
        game.player.y = 0
        # Try moving up and left into negative indices
        game.player.move(0, -1, game.field)
        self.assertEqual(game.player.x, 0)
        wall_penalty = getattr(game.field, "wall_penalty", 50.0)
        self.assertEqual(game.field.score_board["reward"], -wall_penalty)

        game.player.move(-1, 0, game.field)
        self.assertEqual(game.player.x, 0)
        self.assertEqual(game.player.y, 0)
        self.assertEqual(game.field.score_board["reward"], -wall_penalty * 2)

    def test_apple_replenishment_headless(self):
        """Verifies apples replenish up to apple_max during headless training steps."""
        config = create_default_config()
        config.step_delay = 0.0
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)
        env.reset(seed=42)

        self.assertEqual(game.field.score_board["apple_count"], game.field.score_board["apple_max"])

        # Execute 50 steps; apple count should remain at apple_max
        for _ in range(50):
            env.step(np.random.randint(0, 4))
            self.assertEqual(game.field.score_board["apple_count"], game.field.score_board["apple_max"])

    def test_no_artificial_truncation_penalty(self):
        """Verifies truncation at max_step does not inject -100 penalty."""
        config = create_default_config()
        config.step_delay = 0.0
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config)
        env.reset()
        env.max_step = 2

        obs, r1, term1, trunc1, _ = env.step(0)
        self.assertFalse(trunc1)
        obs, r2, term2, trunc2, _ = env.step(0)
        self.assertTrue(trunc2)
        # Reward should be standard step reward (-0.1 or collision -10.1), NOT <= -100
        self.assertGreater(r2, -50.0)

    def test_graceful_stop_callback(self):
        """Verifies threading stop event stops SB3 training cleanly."""
        import threading
        from rl_game.engine.game import GracefulStopCallback

        stop_event = threading.Event()
        callback = GracefulStopCallback(stop_event)
        self.assertTrue(callback._on_step())
        stop_event.set()
        self.assertFalse(callback._on_step())

    def test_open_closed_principle_extensibility(self):
        """Verifies custom registered algorithm can be used via RLConfig."""
        from rl_game.rl.factory import RLAlgorithmStrategy
        from stable_baselines3.common.base_class import BaseAlgorithm

        class DummyStrategy(RLAlgorithmStrategy):
            def build_model(self, config, env) -> BaseAlgorithm:
                return "DUMMY_MODEL"

        RLModelFactory.register("CUSTOM_ALGO", DummyStrategy())
        cfg = RLConfig(algorithm="CUSTOM_ALGO")
        model = RLModelFactory.create(cfg, None)
        self.assertEqual(model, "DUMMY_MODEL")

    def test_bad_apple_env_wall_termination(self):
        """Verifies that BadAppleEnv terminates on wall collision when terminate_on_wall=True."""
        game = BadAppleGame(width=10, height=10, cell_size=20)
        # Player is at (5, 3). Moving UP (action 0) 3 times hits the top wall (y=0)
        env = BadAppleEnv(game=game, terminate_on_wall=True)
        env.reset()
        # Step 1: to (5, 2)
        _, _, term, trunc, _ = env.step(0)
        self.assertFalse(term)
        self.assertFalse(trunc)
        # Step 2: to (5, 1)
        _, _, term, trunc, _ = env.step(0)
        self.assertFalse(term)
        # Step 3: to (5, 0) - top wall collision!
        _, _, term, trunc, _ = env.step(0)
        self.assertTrue(term)  # Must be Game Over!

    def test_grpo_infer_until_game_over_and_train_all_at_once(self):
        """Verifies that GRPO runs inference until terminated or truncated, then trains all at once."""
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            grpo=GRPOConfig(
                group_size=1,
                rollout_steps=None,  # Infer until game over or time over
                batch_size=8,
                n_epochs=2,
                verbose=0,
            ),
        )
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=config, terminate_on_wall=True)
        model = RLModelFactory.create(config, env)

        # Collect 1 completed episode (it will infer until wall collision or timeout)
        trajectories, steps, cont = model._collect_group_rollouts()
        self.assertTrue(cont)
        self.assertEqual(len(trajectories), 1)
        self.assertGreater(steps, 0)
        # Trajectory must have terminated (wall collision) or truncated (timeout)
        self.assertTrue(trajectories[0]["terminated"] or trajectories[0]["truncated"])
        # Verify advantages are computed
        self.assertEqual(trajectories[0]["advantages"].shape[0], steps)

        # Train 1 iteration (must train all at once on that completed episode)
        model.learn(total_timesteps=steps + 1)
        self.assertGreaterEqual(model.num_timesteps, steps)

    def test_grpo_gradient_accumulation_and_batch_epochs(self):
        """Verifies that GRPO trains with group_size >= 2, gradient accumulation, and 4 epochs."""
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            temporal=TemporalConfig(
                enabled=True,
                n_frames=2,
                temporal_type=TemporalType.TRANSFORMER,
                hidden_dim=32,
                features_dim=32,
                spatial_channels=(16, 32),
                n_heads=2,
                n_layers=1,
            ),
            grpo=GRPOConfig(
                group_size=2,
                rollout_steps=None,
                batch_size=4,
                n_epochs=4,
                gradient_accumulation_steps=2,
                verbose=0,
            ),
        )
        game = BadAppleGame(width=8, height=8, cell_size=15)
        env = BadAppleEnv(game=game, config=config, terminate_on_wall=True)
        model = RLModelFactory.create(config, env)

        self.assertEqual(model.group_size, 2)
        self.assertEqual(model.gradient_accumulation_steps, 2)
        self.assertEqual(model.n_epochs, 4)

        # Collect 2 episodes
        trajectories, steps, cont = model._collect_group_rollouts()
        self.assertTrue(cont)
        self.assertEqual(len(trajectories), 2)
        self.assertGreater(steps, 0)

        # Learn with gradient accumulation
        model.learn(total_timesteps=steps + 1)
        self.assertGreaterEqual(model.num_timesteps, steps)

    def test_grpo_step_discounted_advantage_computation(self):
        """Verifies step-discounted advantage mode produces time-varying step advantages across rollouts."""
        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            grpo=GRPOConfig(
                group_size=2,
                rollout_steps=None,
                batch_size=4,
                n_epochs=2,
                advantage_mode="step_discounted",
                verbose=0,
            ),
        )
        game = BadAppleGame(width=8, height=8, cell_size=15)
        env = BadAppleEnv(game=game, config=config, terminate_on_wall=True)
        model = RLModelFactory.create(config, env)

        trajectories, steps, cont = model._collect_group_rollouts()
        self.assertTrue(cont)
        self.assertEqual(len(trajectories), 2)
        for traj in trajectories:
            adv = traj["advantages"].cpu().numpy()
            self.assertFalse(np.isnan(adv).any())
            self.assertFalse(np.isinf(adv).any())
            # For trajectory with length > 1, step-discounted advantages vary with time-to-go
            if len(adv) > 1:
                self.assertFalse(np.all(adv == adv[0]))

    def test_goal_and_difficulty_scaling_only_on_actual_clear(self):
        """Verifies that wall collision death does NOT trigger goal count or reduce max_step.
        Only actual game clear (score >= 1000) increments goal_count, and reaching 10 clears reduces max_step.
        """
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, terminate_on_wall=True)
        env.reset()

        self.assertEqual(env.max_step, 1000)
        self.assertEqual(env.goal_count, 0)
        self.assertFalse(env.is_goal)

        # 1. Trigger wall collision (death)
        # Move up 3 times to hit wall
        env.step(0)
        env.step(0)
        _, _, term, _, _ = env.step(0)
        self.assertTrue(term)
        # Death should NOT increment goal_count or change max_step
        self.assertEqual(env.goal_count, 0)
        self.assertEqual(env.max_step, 1000)
        self.assertFalse(env.is_goal)

        # Resetting after death should not trigger is_goal wall recoloring
        env.reset()
        self.assertFalse(env.is_goal)

        # 2. Simulate 9 game clears (score = 1000)
        for i in range(1, 10):
            env.game.field.score_board["score"] = 1000
            env.goal()
            self.assertEqual(env.goal_count, i)
            self.assertEqual(env.max_step, 1000)
            self.assertTrue(env.is_goal)

        # 3. 10th game clear triggers Level Up (reduces max_step by 100, resets goal_count)
        env.game.field.score_board["score"] = 1000
        env.goal()
        self.assertEqual(env.goal_count, 0)
        self.assertEqual(env.max_step, 900)
        self.assertEqual(env.game.field.score_board["max_step"], 900)
        self.assertEqual(env.total_clears, 10)

    def test_transformer_architecture_and_forward_inference(self):
        """Verifies default model uses TemporalTransformerFeaturesExtractor and produces valid logits."""
        cfg = create_default_config()
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=cfg)
        model = RLModelFactory.create(cfg, env)

        from rl_game.rl.features_extractor import TemporalTransformerFeaturesExtractor
        self.assertIsInstance(model.actor.features_extractor, TemporalTransformerFeaturesExtractor)
        self.assertEqual(model.actor.features_extractor.d_model, 128)
        self.assertEqual(model.actor.features_extractor.transformer.num_layers, 2)

        # Single prediction inference
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        self.assertIn(action, [0, 1, 2, 3])

    def test_full_episode_sequence_transformer_causal_forward(self):
        """Verifies that Full-Episode Transformer processes complete trajectory sequences with causal masking."""
        cfg = create_default_config()
        cfg.grpo.group_size = 2
        cfg.step_delay = 0.0
        game = BadAppleGame(width=8, height=8, cell_size=15)
        env = BadAppleEnv(game=game, config=cfg, terminate_on_wall=True)
        model = RLModelFactory.create(cfg, env)

        # 1. Verify observation space is single frame (raw_channels, H, W) = (4, 8, 8)
        self.assertEqual(env.observation_space.shape, (4, 8, 8))

        # 2. Collect 2 complete episodes using step_inference with token history
        trajectories, steps, cont = model._collect_group_rollouts()
        self.assertTrue(cont)
        self.assertEqual(len(trajectories), 2)
        self.assertGreater(steps, 0)

        # 3. Test forward_trajectory on each episode
        for traj in trajectories:
            obs = torch.as_tensor(traj["obs"])
            L = len(obs)
            dist = model.actor.forward_trajectory(obs)
            self.assertEqual(dist.logits.shape, (L, 4))
            actions = torch.as_tensor(traj["actions"])
            log_probs = dist.log_prob(actions)
            self.assertEqual(log_probs.shape, (L,))

        # 4. Run learn() on the full trajectories
        model.learn(total_timesteps=steps + 1)
        self.assertGreaterEqual(model.num_timesteps, steps)


if __name__ == "__main__":
    unittest.main()
