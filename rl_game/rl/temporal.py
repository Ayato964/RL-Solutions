from collections import deque
from typing import Deque, Optional
import numpy as np


class TemporalObservationBuffer:
    """A ring buffer maintaining a sliding window of historical observations.

    Guarantees O(1) frame appending and memory-efficient concatenation along
    the channel axis for temporal models and frame stacking.
    """

    def __init__(self, n_frames: int, enabled: bool = True):
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")
        self.n_frames = n_frames
        self.enabled = enabled
        self._buffer: Deque[np.ndarray] = deque(maxlen=n_frames if enabled else 1)

    @property
    def is_enabled(self) -> bool:
        return self.enabled

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """Resets the buffer by filling all temporal slots with the initial observation."""
        self._buffer.clear()
        target_count = self.n_frames if self.enabled else 1
        for _ in range(target_count):
            self._buffer.append(initial_frame)
        return self.get_observation()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """Appends a new observation frame to the sliding window and returns the stacked state."""
        self._buffer.append(frame)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Returns the current temporal observation array.

        If temporal is enabled and n_frames > 1, stacks frames along axis 0
        (channels): shape (n_frames * C, H, W).
        Otherwise returns the single frame: shape (C, H, W).
        """
        if not self._buffer:
            raise RuntimeError("Temporal buffer is empty. Call reset() before accessing observations.")

        if not self.enabled or self.n_frames == 1:
            return self._buffer[-1]

        # Stack along channel axis (axis 0: channels)
        return np.concatenate(list(self._buffer), axis=0)
