from __future__ import annotations
import numpy as np
import gymnasium as gym

ENV_ID = "LunarLander-v3"
MODES = ("none", "time_penalty", "hover_hack", "fixed")


def _speed(obs) -> float:
    return float(np.hypot(obs[2], obs[3]))


def _upright(obs) -> float:
    return abs(float(obs[4]))


def niceness(obs) -> float:
    """Continuous 'how upright and slow am I' score in [0, 2]. Used as potential for `fixed`."""
    up = 1.0 - min(_upright(obs) / 0.2, 1.0)
    slow = 1.0 - min(_speed(obs) / 0.3, 1.0)
    return float(up + slow)


def is_nice(obs) -> bool:
    """Binary version used by the (bad) hover_hack rule."""
    return _upright(obs) < 0.2 and _speed(obs) < 0.3


class ShapedReward(gym.Wrapper):
    """Adds custom reward terms on top of the built-in reward.

    Dynamics are untouched; only the reward changes. Per-step `info` gets
    `raw_reward`, `bonus`; on episode end `info['ep_stats']` is filled with
    raw_return, shaped_return, length and outcome (landed / crashed / timeout).
    """

    def __init__(self, env, mode="none", time_fee=0.05, hover_bonus=1.0, gamma=0.999):
        super().__init__(env)
        assert mode in MODES, f"Unknown mode: {mode}. Allowed: {MODES}"
        self.mode = mode
        self.time_fee = float(time_fee)
        self.hover_bonus = float(hover_bonus)
        self.gamma = float(gamma)  # must match PPO's gamma for the policy-invariance argument
        self._raw = 0.0
        self._shaped = 0.0
        self._len = 0
        self._phi = 0.0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._raw = 0.0
        self._shaped = 0.0
        self._len = 0
        self._phi = niceness(obs)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        bonus = 0.0
        potential_bonus = 0.0
        time_fee = 0.0

        if self.mode == "time_penalty":
            time_fee = self.time_fee
            bonus = -time_fee
        elif self.mode == "hover_hack":
            bonus = self.hover_bonus if (not term and is_nice(obs)) else 0.0
        elif self.mode == "fixed":
            phi_new = niceness(obs)
            phi_next = 0.0 if (term or trunc) else phi_new
            potential_bonus = self.gamma * phi_next - self._phi
            time_fee = self.time_fee
            bonus = potential_bonus - time_fee
            self._phi = phi_next

        shaped = r + bonus
        self._raw += r
        self._shaped += shaped
        self._len += 1

        info = dict(info, raw_reward=r, bonus=bonus,
                    potential_bonus=potential_bonus, time_fee=time_fee)

        if term or trunc:
            if trunc and not term:
                outcome = "timeout"
            elif r >= 100 - 1e-6:
                outcome = "landed"
            else:
                outcome = "crashed"  # includes out-of-bounds (-100)

            info["ep_stats"] = dict(
                raw_return=self._raw,
                shaped_return=self._shaped,
                length=self._len,
                outcome=outcome,
            )

        return obs, shaped, term, trunc, info


def make_env(mode="none", render_mode=None, **kw):
    env = gym.make(ENV_ID, render_mode=render_mode)
    return env if mode == "raw" else ShapedReward(env, mode=mode, **kw)


def make_raw_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)

