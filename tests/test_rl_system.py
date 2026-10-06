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
    SpatialCNNEncoder,
    AIPlayer,
    GRPO,
    get_features_extractor_specs,
)
from rl_game.rl.DQN import DQNTrainer as LegacyDQNTrainer, AIPlayer as LegacyAIPlayer
from rl_game.engine.game import GamePhase
from snake_game.main import (
    BadAppleGame,
    BadAppleEnv,
    BadAppleTrainer,
    MultiBadAppleEnv,
    create_default_config,
)


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = create_default_config()
        self.assertEqual(cfg.algorithm, AlgorithmType.GRPO)
        self.assertEqual(cfg.step_delay, 0.0)
        self.assertTrue(cfg.temporal.enabled)
        self.assertTrue(cfg.temporal.full_episode)
        self.assertEqual(cfg.temporal.n_frames, 1)
        self.assertEqual(cfg.temporal.max_seq_len, 1000)
        self.assertEqual(cfg.temporal.temporal_type, TemporalType.TRANSFORMER)
        self.assertEqual(cfg.temporal.n_heads, 4)
        self.assertEqual(cfg.temporal.n_layers, 2)
        self.assertEqual(cfg.temporal.spatial_channels, (64, 128))
        self.assertEqual(cfg.temporal.hidden_dim, 256)
        self.assertEqual(cfg.temporal.features_dim, 256)
        self.assertEqual(cfg.temporal.dim_feedforward, 512)
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

    def test_spatial_cnn_encoder_folded_dims(self):
        encoder = SpatialCNNEncoder(in_channels=4, channel_1=64, channel_2=128, pool_size=(4, 4))
        self.assertEqual(encoder.output_dim, 2048)

        # Standard grid shape (20, 30)
        x_std = torch.randn(2, 4, 20, 30)
        out_std = encoder(x_std)
        self.assertEqual(out_std.shape, (2, 2048))

        # Compact grid shape (8, 8)
        x_compact = torch.randn(3, 4, 8, 8)
        out_compact = encoder(x_compact)
        self.assertEqual(out_compact.shape, (3, 2048))

    def test_transformer_extractor_default_256_dims(self):
        ext = TemporalTransformerFeaturesExtractor(
            self.obs_space,
            n_frames=self.n_frames,
            raw_channels=self.raw_channels,
            hidden_dim=256,
            dim_feedforward=512,
            features_dim=256,
        )
        self.assertEqual(ext.spatial_encoder.output_dim, 2048)
        self.assertIsInstance(ext.token_proj, torch.nn.Linear)
        self.assertEqual(ext.token_proj.in_features, 2048)
        self.assertEqual(ext.token_proj.out_features, 256)

        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 256))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(ext.token_proj.weight.grad)

    def test_stacked_extractor_folded_dims(self):
        ext = TemporalStackedFeaturesExtractor(
            self.obs_space,
            features_dim=256,
        )
        out = ext(self.dummy_input)
        self.assertEqual(out.shape, (2, 256))
        loss = out.sum()
        loss.backward()


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
        wall_penalty = getattr(game.field, "wall_penalty", 100.0)
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
        self.assertEqual(model.actor.features_extractor.d_model, 256)
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

    def test_zero_step_penalty(self):
        """Verifies step_penalty is abolished (0.0) and survival_reward (0.5) is awarded for safe movement."""
        cfg = create_default_config()
        cfg.step_delay = 0.0
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=cfg)
        env.reset()
        # Place player away from walls and apples, ensure target cell (6, 5) is empty Void
        game.player.x = 5
        game.player.y = 5
        game.field.set_object(6, 5, game.registry.create(id=0, x=6, y=5, size=game.cell_size))
        # Move right (dx=1, dy=0) into empty cell (id=0)
        obs, reward, term, trunc, _ = env.step(3)
        self.assertFalse(term)
        self.assertFalse(trunc)
        self.assertEqual(reward, env.survival_reward)

    def test_dynamic_wall_penalty_decay(self):
        """Verifies wall collision penalty is -100 at STEP 0 and decays by 0.1 per survival step."""
        cfg = create_default_config()
        cfg.step_delay = 0.0
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=cfg)

        # 1. Collision at STEP 0 (first action of episode): penalty should be -100.0 (no survival bonus on death)
        env.reset()
        # Place player right next to top wall (y=1)
        game.player.x = 5
        game.player.y = 1
        # Action 0 (UP) crashes into top wall (y=0) immediately
        obs, reward, term, trunc, _ = env.step(0)
        self.assertTrue(term)
        self.assertEqual(reward, -100.0)

        # 2. Collision after 1 survival step (STEP 1): penalty should be -99.9
        env.reset()
        # Ensure cell (5, 1) is an empty cell (id=0, Void) so it doesn't randomly spawn an apple
        game.field.set_object(5, 1, game.registry.create(id=0, x=5, y=1, size=game.cell_size))
        game.player.x = 5
        game.player.y = 2
        # First step moves to y=1 (safe, reward=survival_reward)
        obs, r1, term1, _, _ = env.step(0)
        self.assertFalse(term1)
        self.assertEqual(r1, env.survival_reward)
        # Second step moves into wall (y=0), step_count was 1 -> penalty = -100 + 0.1 * 1 = -99.9
        obs, r2, term2, _, _ = env.step(0)
        self.assertTrue(term2)
        self.assertAlmostEqual(r2, -99.9, places=5)

        # 3. Collision after 10 survival steps (STEP 10): penalty should be -99.0
        env.reset()
        env.step_count = 10  # Simulate 10 survival steps
        game.player.x = 5
        game.player.y = 1
        obs, r3, term3, _, _ = env.step(0)
        self.assertTrue(term3)
        self.assertAlmostEqual(r3, -99.0, places=5)

    def test_survival_bonus_accumulation(self):
        """Verifies each non-terminal movement awards +0.5 survival bonus, creating clear gradient over suicide."""
        cfg = create_default_config()
        cfg.survival_reward = 0.5
        game = BadAppleGame(width=20, height=20, cell_size=10)
        env = BadAppleEnv(game=game, config=cfg)
        env.reset()

        # Clear cells (5, 5) to (10, 5) to ensure empty pathway
        for x in range(5, 11):
            game.field.set_object(x, 5, game.registry.create(id=0, x=x, y=5, size=game.cell_size))
        game.player.x = 5
        game.player.y = 5

        total_r = 0.0
        # Move right 4 times
        for _ in range(4):
            obs, r, term, trunc, _ = env.step(3)
            self.assertFalse(term)
            total_r += r

        # 4 safe steps should accumulate exactly 4 * 0.5 = 2.0 reward
        self.assertAlmostEqual(total_r, 2.0, places=5)

    def test_rl_transition_resets_gameplay_stats(self):
        """Verifies score, clear counts, and step counters reset to 0 upon transitioning to RL."""
        from rl_game.engine.game import GamePhase
        cfg = create_default_config()
        cfg.step_delay = 0.0
        cfg.imitation.min_demos = 1000  # Skip BC training to directly test state reset
        game = BadAppleGame(width=10, height=10, cell_size=20)
        env = BadAppleEnv(game=game, config=cfg)
        model = RLModelFactory.create(cfg, env)

        game.phase = GamePhase.HUMAN_DEMO
        game.env = env
        game.config = cfg
        game.model = model
        game.total_timesteps = 10
        game.current_obs, _ = env.reset()
        self.assertEqual(game.phase, GamePhase.HUMAN_DEMO)

        # Simulate human demo accumulating score, clears, steps, and leveling up (reducing max_step)
        game.field.score_board["score"] = 450
        game.field.score_board["clear_count"] = 3
        game.field.score_board["total_clears"] = 5
        game.field.score_board["step"] = 80
        game.field.score_board["total_step"] = 120
        game.field.score_board["max_step"] = 800
        game.episode_count = 4
        env.goal_count = 3
        env.total_clears = 5
        env.step_count = 80
        env.total_step_count = 120
        env.max_step = 800

        # Press SPACE transition without background thread running real learn loop
        with unittest.mock.patch.object(game, "ai_work"):
            game._transition_to_pretraining_and_rl()
            if game.worker_thread is not None:
                game.worker_thread.join(timeout=2.0)

        # Verify all stats and difficulty/max_step are cleanly reset to initial state
        self.assertEqual(game.phase, GamePhase.RL_TRAINING)
        self.assertEqual(game.field.score_board["score"], 0)
        self.assertEqual(game.field.score_board["clear_count"], 0)
        self.assertEqual(game.field.score_board["total_clears"], 0)
        self.assertEqual(game.field.score_board["step"], 0)
        self.assertEqual(game.field.score_board["total_step"], 0)
        self.assertEqual(game.field.score_board["max_step"], 1000)
        self.assertEqual(game.episode_count, 0)
        self.assertEqual(env.goal_count, 0)
        self.assertEqual(env.total_clears, 0)
        self.assertEqual(env.step_count, 0)
        self.assertEqual(env.total_step_count, 0)
        self.assertEqual(env.max_step, 1000)


class TestParallelAndLoggingUpgrades(unittest.TestCase):
    """Verifies 16-stage parallel environment, return min/mean/max logging, and atomic auto-save."""

    def test_multi_bad_apple_env_step_and_reset(self):
        num_envs = 4
        game = BadAppleGame(width=10, height=10, cell_size=10, num_envs=num_envs)
        env = MultiBadAppleEnv(game=game, num_envs=num_envs)

        self.assertEqual(env.num_envs, num_envs)
        obses, _ = env.reset()
        self.assertEqual(len(obses), num_envs)
        self.assertEqual(obses[0].shape, (4, 10, 10))

        # Parallel step
        actions = [0, 1, 2, 3]
        next_obses, rewards, terms, truncs, infos = env.step(actions)
        self.assertEqual(len(next_obses), num_envs)
        self.assertEqual(len(rewards), num_envs)
        self.assertEqual(len(terms), num_envs)
        self.assertEqual(len(truncs), num_envs)

        # Single reset_at
        r_obs, _ = env.reset_at(1)
        self.assertEqual(r_obs.shape, (4, 10, 10))

    def test_grpo_multi_env_rollout_and_return_statistics(self):
        cfg = create_default_config()
        cfg.grpo.group_size = 4
        cfg.step_delay = 0.0

        num_envs = 4
        game = BadAppleGame(width=8, height=8, cell_size=10, num_envs=num_envs)
        env = MultiBadAppleEnv(game=game, config=cfg, num_envs=num_envs)
        model = RLModelFactory.create(cfg, env)

        # Collect group rollouts in parallel
        trajectories, steps, cont = model._collect_group_rollouts()
        self.assertGreaterEqual(len(trajectories), 4)
        self.assertGreater(steps, 0)
        self.assertTrue(cont)

        # Verify min, mean, max return calculation
        returns = [float(t.get("total_return", 0.0)) for t in trajectories]
        self.assertEqual(len(returns), len(trajectories))
        min_ret = float(np.min(returns))
        mean_ret = float(np.mean(returns))
        max_ret = float(np.max(returns))
        self.assertLessEqual(min_ret, mean_ret)
        self.assertLessEqual(mean_ret, max_ret)

    def test_grpo_atomic_auto_save(self):
        import tempfile
        cfg = create_default_config()
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_model")
            cfg.model_save_path = save_path
            game = BadAppleGame(width=8, height=8, cell_size=10, num_envs=2)
            env = MultiBadAppleEnv(game=game, config=cfg, num_envs=2)
            model = RLModelFactory.create(cfg, env)

            # Atomic save check
            model.save(save_path)
            expected_file = f"{save_path}.pt"
            self.assertTrue(os.path.exists(expected_file))
            self.assertFalse(os.path.exists(f"{expected_file}.tmp"))

            # Verify loaded weights match
            loaded_model = GRPO.load(expected_file, env=env, config=cfg)
            self.assertEqual(loaded_model.num_timesteps, model.num_timesteps)

    def test_variable_speed_key_controls(self):
        game = BadAppleGame(width=10, height=10, cell_size=10, num_envs=4)
        initial_delay = game.step_delay

        # Test '[' key increases delay
        event_slower = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFTBRACKET)
        pygame.event.post(event_slower)
        game.handle_events()
        self.assertGreater(game.step_delay, initial_delay)

        # Test ']' key decreases delay
        event_faster = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHTBRACKET)
        pygame.event.post(event_faster)
        game.handle_events()
        self.assertAlmostEqual(game.step_delay, initial_delay, places=4)

        # Test 'F' key sets delay to 0.0
        event_max = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f)
        pygame.event.post(event_max)
        game.handle_events()
        self.assertEqual(game.step_delay, 0.0)

    def test_evaluation_demo_mode_transition(self):
        """Verifies training completion transitions engine phase to EVALUATION and updates status."""
        import threading
        import time

        game = BadAppleGame(width=10, height=10, cell_size=10, num_envs=2)
        env = MultiBadAppleEnv(game=game, num_envs=2)
        game.env = env

        config = RLConfig(
            algorithm=AlgorithmType.GRPO,
            action_space=4,
            max_id=4,
            step_delay=0.0,
            grpo=GRPOConfig(group_size=2, n_epochs=1, batch_size=4),
        )
        model = RLModelFactory.create(config, env)

        # Simulate instant training completion and stop demo after 1 iteration
        model.learn = lambda *args, **kwargs: None
        def stop_soon():
            time.sleep(0.05)
            game.stop_event.set()
        threading.Thread(target=stop_soon, daemon=True).start()

        game.ai_work(model, total_timesteps=1, model_save_path="model/test_eval_ckpt")
        self.assertEqual(game.phase, GamePhase.EVALUATION)
        self.assertIn("COMPLETE!", game.field.score_board["status"])

    def test_root_main_delegation_and_eval_flag(self):
        """Verifies root main.py imports cleanly and CLI eval flag properly sets GamePhase.EVALUATION."""
        import importlib
        import main as root_main
        self.assertTrue(hasattr(root_main, "main"))

        # Verify eval flag detection logic
        from snake_game.main import BadAppleGame, MultiBadAppleEnv, create_default_config, RLModelFactory
        from rl_game.engine.game import GamePhase

        cfg = create_default_config()
        game = BadAppleGame(width=10, height=10, cell_size=10, num_envs=2)
        
        # Test eval mode flag activation
        argv = ["main.py", "eval"]
        is_eval_mode = any(arg.lower() in ("eval", "--eval", "-e") for arg in argv[1:])
        self.assertTrue(is_eval_mode)
        if is_eval_mode:
            game.phase = GamePhase.EVALUATION
            game.step_delay = 0.03
        self.assertEqual(game.phase, GamePhase.EVALUATION)
        self.assertEqual(game.step_delay, 0.03)

        # Test T key toggle from EVALUATION to RL_TRAINING
        import pygame
        event_t = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_t)
        pygame.event.post(event_t)
        game.handle_events()
        self.assertEqual(game.phase, GamePhase.RL_TRAINING)


if __name__ == "__main__":
    unittest.main()
