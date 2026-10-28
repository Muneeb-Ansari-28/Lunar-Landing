"""Unit and invariant tests for ShapedReward wrapper in src/ll/envs.py."""
import gymnasium as gym
import numpy as np
import pytest
from ll.envs import ShapedReward, niceness, is_nice, ENV_ID

def test_mode_none_identity():
    """mode='none' rewards must equal raw env rewards exactly for same seed/actions."""
    raw_env = gym.make(ENV_ID)
    shaped_env = ShapedReward(gym.make(ENV_ID), mode="none")

    raw_obs, _ = raw_env.reset(seed=123)
    shaped_obs, _ = shaped_env.reset(seed=123)
    np.testing.assert_allclose(raw_obs, shaped_obs)

    for step in range(50):
        action = step % 4
        r_obs, r_reward, r_term, r_trunc, _ = raw_env.step(action)
        s_obs, s_reward, s_term, s_trunc, s_info = shaped_env.step(action)

        np.testing.assert_allclose(r_obs, s_obs)
        assert r_reward == pytest.approx(s_reward, abs=1e-7)
        assert s_info["bonus"] == 0.0
        assert r_term == s_term
        assert r_trunc == s_trunc
        if r_term or r_trunc:
            break

    raw_env.close()
    shaped_env.close()


def test_hover_hack_bonus_bounds():
    """hover_hack gives <= +1 bonus per frame and 0 on terminal frame."""
    env = ShapedReward(gym.make(ENV_ID), mode="hover_hack", hover_bonus=1.0)
    obs, _ = env.reset(seed=42)

    for _ in range(200):
        action = env.action_space.sample()
        obs, shaped_r, term, trunc, info = env.step(action)
        bonus = info["bonus"]
        assert 0.0 <= bonus <= 1.0 + 1e-7
        if term:
            # On terminal frame, hover_hack bonus must be 0
            assert bonus == 0.0
            break

    env.close()


def test_fixed_potential_bounded_telescoping():
    """Fixed mode potential bonus cannot farm reward: cumulative potential <= 0 + eps."""
    gamma = 0.999
    time_fee = 0.05
    env = ShapedReward(gym.make(ENV_ID), mode="fixed", gamma=gamma, time_fee=time_fee)
    obs, _ = env.reset(seed=999)

    cum_potential = 0.0
    cum_time_fee = 0.0
    cum_bonus = 0.0
    length = 0

    done = False
    while not done:
        action = 0  # Noop: falling/tumbling/landing/crashing
        obs, shaped_r, term, trunc, info = env.step(action)
        cum_potential += info["potential_bonus"]
        cum_time_fee += info["time_fee"]
        cum_bonus += info["bonus"]
        length += 1
        done = term or trunc

    # Over the entire episode, next-state potential is 0 at termination.
    # Telescoping sum: gamma * Phi(s_T=0) - Phi(s_0) + (gamma-1) * sum(Phi) <= 0
    assert cum_potential <= 0.0 + 1e-5, f"Cumulative potential farmed points: {cum_potential}"
    assert cum_time_fee == pytest.approx(time_fee * length, abs=1e-5)
    assert cum_bonus == pytest.approx(cum_potential - cum_time_fee, abs=1e-5)
    assert cum_bonus <= -time_fee * length + 1e-5

    env.close()


def test_timeout_outcome_detection():
    """ep_stats['outcome'] must be 'timeout' when truncated without terminal condition."""
    env = ShapedReward(gym.make(ENV_ID), mode="time_penalty")
    obs, _ = env.reset(seed=100)

    # Mock step to simulate truncation at 1000 frames
    mock_obs = np.zeros(8, dtype=np.float32)
    mock_reward = 0.0
    mock_term = False
    mock_trunc = True
    mock_info = {}

    env.env.step = lambda a: (mock_obs, mock_reward, mock_term, mock_trunc, mock_info)
    _, _, _, _, info = env.step(0)

    assert "ep_stats" in info
    assert info["ep_stats"]["outcome"] == "timeout"
    env.close()

