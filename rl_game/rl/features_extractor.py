from typing import Any, Dict, Optional, Tuple, Type
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl_game.rl.config import TemporalConfig, TemporalType


class SpatialCNNEncoder(nn.Module):
    """Encodes a single 2D grid observation into a 2048-dim spatial feature vector (128 channels x 4x4).

    Features deeper receptive field CNN layers folding spatial features into 128 channels x 4x4,
    avoiding severe channel bottleneck (16 channels) and high-dimensional flattening (9600 dim).
    """

    def __init__(
        self,
        in_channels: int,
        channel_1: int = 64,
        channel_2: int = 128,
        channel_out: Optional[int] = None,
        grid_shape: Optional[Tuple[int, int]] = (20, 30),
        pool_size: Tuple[int, int] = (4, 4),
        **kwargs,
    ):
        super().__init__()
        layers = [
            # Layer 1: Low-level spatial entity detection
            nn.Conv2d(in_channels, channel_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 2: Intermediate composite feature mapping
            nn.Conv2d(channel_1, channel_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Layer 3: Spatial folding with stride 2 downsampling (expands receptive field)
            nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Layer 4: Deep feature refinement maintaining 128 channels
            nn.Conv2d(channel_2, channel_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # Pooling layer ensuring exact 4x4 spatial resolution across any input grid size
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
        ]
        self.conv = nn.Sequential(*layers)
        self.output_dim = channel_2 * pool_size[0] * pool_size[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalLSTMFeaturesExtractor(BaseFeaturesExtractor):
    """Time-series feature extractor using a Spatial CNN followed by an LSTM sequence model.

    Observations: (Batch, n_frames * raw_channels, Height, Width)
    1. Reshapes to (Batch * n_frames, raw_channels, Height, Width) and processes in parallel via Spatial CNN.
    2. Reshapes back to sequence (Batch, n_frames, spatial_dim).
    3. Feeds sequence into an LSTM.
    4. Projects the final temporal hidden state into features_dim.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_frames: int = 4,
        raw_channels: int = 4,
        hidden_dim: int = 256,
        features_dim: int = 256,
        spatial_channels: Tuple[int, int] = (64, 128),
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.n_frames = n_frames
        self.raw_channels = raw_channels

        grid_shape = (
            (observation_space.shape[1], observation_space.shape[2])
            if hasattr(observation_space, "shape") and len(observation_space.shape) >= 3
            else (20, 30)
        )
        self.spatial_encoder = SpatialCNNEncoder(
            in_channels=raw_channels,
            channel_1=spatial_channels[0],
            channel_2=spatial_channels[1],
            grid_shape=grid_shape,
        )

        self.lstm = nn.LSTM(
            input_size=self.spatial_encoder.output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, _, h, w = observations.shape
        t = self.n_frames
        c = self.raw_channels

        # Reshape to (Batch * Time, Channels, Height, Width) for fast parallel GPU/CPU encoding
        x = observations.float().reshape(b * t, c, h, w)
        spatial_feats = self.spatial_encoder(x)  # (B * T, spatial_dim)

        # Reshape back to temporal sequence: (Batch, Time, spatial_dim)
        seq = spatial_feats.reshape(b, t, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(seq)  # (Batch, Time, hidden_dim)

        # Output from the most recent timestep
        last_step = lstm_out[:, -1, :]  # (Batch, hidden_dim)
        return self.fc(last_step)


class TemporalGRUFeaturesExtractor(BaseFeaturesExtractor):
    """Time-series feature extractor using a Spatial CNN followed by a GRU sequence model."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_frames: int = 4,
        raw_channels: int = 4,
        hidden_dim: int = 256,
        features_dim: int = 256,
        spatial_channels: Tuple[int, int] = (64, 128),
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.n_frames = n_frames
        self.raw_channels = raw_channels

        grid_shape = (
            (observation_space.shape[1], observation_space.shape[2])
            if hasattr(observation_space, "shape") and len(observation_space.shape) >= 3
            else (20, 30)
        )
        self.spatial_encoder = SpatialCNNEncoder(
            in_channels=raw_channels,
            channel_1=spatial_channels[0],
            channel_2=spatial_channels[1],
            grid_shape=grid_shape,
        )

        self.gru = nn.GRU(
            input_size=self.spatial_encoder.output_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, _, h, w = observations.shape
        t = self.n_frames
        c = self.raw_channels

        x = observations.float().reshape(b * t, c, h, w)
        spatial_feats = self.spatial_encoder(x)
        seq = spatial_feats.reshape(b, t, -1)

        gru_out, _ = self.gru(seq)
        last_step = gru_out[:, -1, :]
        return self.fc(last_step)


class TemporalConv1dFeaturesExtractor(BaseFeaturesExtractor):
    """Temporal feature extractor using 1D convolution over historical frame embeddings."""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_frames: int = 4,
        raw_channels: int = 4,
        hidden_dim: int = 256,
        features_dim: int = 256,
        spatial_channels: Tuple[int, int] = (64, 128),
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.n_frames = n_frames
        self.raw_channels = raw_channels

        grid_shape = (
            (observation_space.shape[1], observation_space.shape[2])
            if hasattr(observation_space, "shape") and len(observation_space.shape) >= 3
            else (20, 30)
        )
        self.spatial_encoder = SpatialCNNEncoder(
            in_channels=raw_channels,
            channel_1=spatial_channels[0],
            channel_2=spatial_channels[1],
            grid_shape=grid_shape,
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.spatial_encoder.output_dim,
                out_channels=hidden_dim,
                kernel_size=min(3, n_frames),
                padding=1 if n_frames >= 3 else 0,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, _, h, w = observations.shape
        t = self.n_frames
        c = self.raw_channels

        x = observations.float().reshape(b * t, c, h, w)
        spatial_feats = self.spatial_encoder(x)  # (B * T, D)
        seq = spatial_feats.reshape(b, t, -1).transpose(1, 2)  # (B, D, T) for Conv1d

        conv_out = self.temporal_conv(seq.contiguous())  # (B, hidden_dim)
        return self.fc(conv_out)


class TemporalStackedFeaturesExtractor(BaseFeaturesExtractor):
    """Direct multi-channel 2D CNN over concatenated temporal frames."""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        spatial_channels: Tuple[int, int] = (64, 128),
        pool_size: Tuple[int, int] = (4, 4),
        channel_out: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        in_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels[0], spatial_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels[1], spatial_channels[1], kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(spatial_channels[1], spatial_channels[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
        )
        conv_out_dim = spatial_channels[1] * pool_size[0] * pool_size[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        features = self.cnn(x)
        return self.fc(features)


class TemporalTransformerFeaturesExtractor(BaseFeaturesExtractor):
    """Causal Transformer feature extractor for sequential grid observations.

    Pipeline:
    1. Spatial Soft Tokenizer:
       Each 2D grid frame in the temporal window (t=0..T-1) is mapped into a continuous
       d_model 'soft token' via SpatialCNNEncoder and an optional linear projection.
    2. Temporal Positional Embedding:
       Learned temporal positional embeddings are added to preserve chronological frame order.
    3. Causal Multi-Head Self-Attention:
       Transformer encoder with causal attention masking prevents future leakage
       (time step t only attends to frames <= t).
    4. Causal Aggregation Head:
       The contextualized sequence is summarized via the specified aggregation strategy
       ('last' token causal summary, 'mean' pooling, or 'attention' pooling) and projected
       through LayerNorm + Linear into features_dim.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_frames: int = 4,
        raw_channels: int = 4,
        hidden_dim: int = 256,  # d_model
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        features_dim: int = 256,
        spatial_channels: Tuple[int, int] = (64, 128),
        aggregation: str = "last",
        max_seq_len: int = 1000,
        full_episode: bool = True,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.n_frames = n_frames
        self.raw_channels = raw_channels
        self.d_model = hidden_dim
        self.aggregation = aggregation.lower()
        self.max_seq_len = max(max_seq_len, n_frames)
        self.full_episode = full_episode

        # 1. Spatial Soft Tokenizer (Parallel Spatial CNN preserving exact grid resolution)
        grid_shape = (
            (observation_space.shape[1], observation_space.shape[2])
            if hasattr(observation_space, "shape") and len(observation_space.shape) >= 3
            else (20, 30)
        )
        self.spatial_encoder = SpatialCNNEncoder(
            in_channels=raw_channels,
            channel_1=spatial_channels[0],
            channel_2=spatial_channels[1],
            grid_shape=grid_shape,
        )

        # Token projection from CNN output to d_model
        self.token_proj = (
            nn.Linear(self.spatial_encoder.output_dim, self.d_model)
            if self.spatial_encoder.output_dim != self.d_model
            else nn.Identity()
        )

        # 2. Learnable Temporal Positional Embeddings (sized for full episode sequences)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        # 3. Transformer Encoder with Pre-LN (norm_first=True for RL gradient stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Precompute causal mask (lower triangular) for full sequence
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

        # 4. Aggregation Head
        if self.aggregation == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=n_heads,
                batch_first=True,
            )

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, features_dim),
            nn.GELU(),
        )

    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encodes a single board frame into a soft token of shape (B, 1, d_model)."""
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        spatial = self.spatial_encoder(frame.float())
        token = self.token_proj(spatial)
        return token.unsqueeze(1)

    def forward_tokens(self, tokens: torch.Tensor, return_all_steps: bool = False) -> torch.Tensor:
        """Processes token sequence (B, T, d_model) through Causal Self-Attention.

        Args:
            tokens: Soft tokens of shape (B, T, d_model).
            return_all_steps: If True, returns representations for all T steps (B, T, features_dim).
                              If False, returns representation of the last step (B, features_dim).
        """
        b, t, _ = tokens.shape
        pos_tokens = tokens + self.pos_embedding[:, :t, :]
        mask = self.causal_mask[:t, :t]
        causal_tokens = self.transformer(pos_tokens, mask=mask, is_causal=True)

        if return_all_steps:
            flat_tokens = causal_tokens.reshape(b * t, self.d_model)
            out = self.head(flat_tokens)
            return out.reshape(b, t, -1)
        else:
            if self.aggregation == "last":
                summary = causal_tokens[:, -1, :]
            elif self.aggregation == "mean":
                summary = causal_tokens.mean(dim=1)
            elif self.aggregation == "attention":
                query = self.attn_query.expand(b, -1, -1)
                summary, _ = self.pool_attn(query, causal_tokens, causal_tokens)
                summary = summary.squeeze(1)
            else:
                summary = causal_tokens[:, -1, :]
            return self.head(summary)

    def forward_trajectory(self, frames: torch.Tensor) -> torch.Tensor:
        """Processes an entire trajectory sequence of frames through Causal Transformer.

        Args:
            frames: (L, C, H, W) or (B, L, C, H, W)
        Returns:
            Features for all L steps: (L, features_dim) or (B, L, features_dim)
        """
        is_unbatched = (frames.ndim == 4)
        if is_unbatched:
            frames = frames.unsqueeze(0)

        b, l, c, h, w = frames.shape
        frames_flat = frames.float().reshape(b * l, c, h, w)
        spatial = self.spatial_encoder(frames_flat)
        tokens = self.token_proj(spatial).reshape(b, l, self.d_model)

        features = self.forward_tokens(tokens, return_all_steps=True)
        if is_unbatched:
            return features.squeeze(0)
        return features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for Gymnasium / SB3 observation tensors."""
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)

        if observations.ndim == 4:
            b, total_c, h, w = observations.shape
            c = self.raw_channels
            if total_c == c:
                t = 1
                x = observations.float()
            elif total_c % c == 0:
                t = total_c // c
                x = observations.float().reshape(b * t, c, h, w)
            else:
                raise ValueError(f"Cannot divide total_channels {total_c} by raw_channels {c}")
            spatial = self.spatial_encoder(x)
            tokens = self.token_proj(spatial).reshape(b, t, self.d_model)
            return self.forward_tokens(tokens, return_all_steps=False)
        elif observations.ndim == 5:
            b, t, c, h, w = observations.shape
            return self.forward_trajectory(observations)[:, -1, :]
        else:
            raise ValueError(f"Expected 3D, 4D, or 5D observation tensor, got {observations.shape}")


class AIPlayer(BaseFeaturesExtractor):
    """Legacy feature extractor preserved for full backward compatibility."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 4):
        super().__init__(observation_space, features_dim=features_dim)
        in_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(2, 2), stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 2), stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.out = nn.Sequential(
            nn.Linear(8 * 4 * 4, 512),
            nn.GELU(),
            nn.Linear(512, features_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn = self.cnn(x.float())
        return self.out(cnn)


def get_features_extractor_specs(
    temporal_config: TemporalConfig,
    raw_channels: int,
) -> Tuple[Type[BaseFeaturesExtractor], Dict[str, Any]]:
    """Factory helper returning (extractor_class, extractor_kwargs) for SB3 policy_kwargs."""
    if not temporal_config.enabled:
        return AIPlayer, {"features_dim": temporal_config.features_dim}

    t_type = temporal_config.temporal_type

    if t_type == TemporalType.TRANSFORMER:
        extractor_cls = TemporalTransformerFeaturesExtractor
    elif t_type == TemporalType.LSTM:
        extractor_cls = TemporalLSTMFeaturesExtractor
    elif t_type == TemporalType.GRU:
        extractor_cls = TemporalGRUFeaturesExtractor
    elif t_type == TemporalType.CONV1D:
        extractor_cls = TemporalConv1dFeaturesExtractor
    elif t_type == TemporalType.STACK:
        extractor_cls = TemporalStackedFeaturesExtractor
    else:
        raise ValueError(f"Unsupported temporal_type: {t_type}")

    kwargs: Dict[str, Any] = {
        "features_dim": temporal_config.features_dim,
    }

    if t_type == TemporalType.TRANSFORMER:
        kwargs.update({
            "n_frames": temporal_config.n_frames,
            "raw_channels": raw_channels,
            "hidden_dim": temporal_config.hidden_dim,
            "n_heads": temporal_config.n_heads,
            "n_layers": temporal_config.n_layers,
            "dim_feedforward": temporal_config.dim_feedforward,
            "dropout": temporal_config.dropout,
            "spatial_channels": temporal_config.spatial_channels,
            "aggregation": temporal_config.aggregation,
            "max_seq_len": getattr(temporal_config, "max_seq_len", 1000),
            "full_episode": getattr(temporal_config, "full_episode", True),
        })
    elif t_type in (TemporalType.LSTM, TemporalType.GRU, TemporalType.CONV1D):
        kwargs.update({
            "n_frames": temporal_config.n_frames,
            "raw_channels": raw_channels,
            "hidden_dim": temporal_config.hidden_dim,
            "spatial_channels": temporal_config.spatial_channels,
        })
    elif t_type == TemporalType.STACK:
        kwargs.update({
            "spatial_channels": temporal_config.spatial_channels,
        })

    return extractor_cls, kwargs
