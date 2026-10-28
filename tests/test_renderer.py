"""Tests for LunarRenderer in src/ll/renderer.py."""
import gymnasium as gym
import numpy as np
import pytest
from ll.envs import ENV_ID
from ll.renderer import LunarRenderer, WORLD_W, WORLD_H

def test_renderer_initialization_and_frame_shape():
    """Renderer initializes headless and produces 1280x720 RGB frame."""
    renderer = LunarRenderer(width=1280, height=720, headless=True)
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=42)

    frame = renderer.render_frame(
        env=env,
        obs=obs,
        action=2,
        config="baseline",
        seed=0,
        frame_idx=10,
        raw_ret=5.0,
        shaped_ret=5.0,
    )

    assert isinstance(frame, np.ndarray)
    assert frame.shape == (720, 1280, 3), f"Expected (720, 1280, 3), got {frame.shape}"
    assert frame.dtype == np.uint8

    env.close()
    renderer.close()


def test_renderer_world_to_screen_mapping():
    """Verify coordinate transformation bounds."""
    renderer = LunarRenderer(width=1280, height=720, headless=True)

    # (0, 0) world is bottom-left
    sx0, sy0 = renderer.world_to_screen(0.0, 0.0)
    assert sx0 == 0
    assert sy0 == 720

    # (WORLD_W, WORLD_H) is top-right
    sx1, sy1 = renderer.world_to_screen(WORLD_W, WORLD_H)
    assert sx1 == 1280
    assert sy1 == 0

    renderer.close()

