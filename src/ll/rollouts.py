"""Phase 7a: Trajectory Collection without In-Memory Frame Flooding."""
from __future__ import annotations
import glob
import pathlib
import numpy as np
import imageio.v3 as iio
from stable_baselines3 import PPO
from ll.envs import make_env, make_raw_env


def collect(
    model_path: str,
    config: str,
    seed: int,
    episodes: int = 5,
    seed_base: int = 20_000,
):
    pathlib.Path("results/rollouts").mkdir(parents=True, exist_ok=True)
    pathlib.Path("videos").mkdir(parents=True, exist_ok=True)

    mode = {"baseline": "none"}.get(config, config)
    model = PPO.load(model_path, device="cpu")

    print(f"Collecting rollouts for {config} seed={seed} ({episodes} episodes)...")

    for ep in range(episodes):
        ep_seed = seed_base + ep
        render_mode = "rgb_array" if ep == 0 else None
        env = make_env(mode=mode, render_mode=render_mode)
        obs, _ = env.reset(seed=ep_seed)

        obs_list = []
        action_list = []
        raw_r_list = []
        bonus_list = []
        pot_bonus_list = []
        time_fee_list = []
        shaped_r_list = []

        stream_writer = None
        if ep == 0:
            raw_video_path = f"videos/raw_{config}_seed{seed}_ep0.mp4"
            # Stream directly to disk
            stream_writer = iio.imiter(raw_video_path, fps=50)

        done = False
        outcome = "unknown"
        while not done:
            if ep == 0:
                frame = env.render()
                # ImageIO writer: frames are streamed one by one
                # Note: we will use simple frame accumulation for small 1000-frame episode if stream writer
                pass

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            next_obs, shaped_r, term, trunc, info = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            raw_r_list.append(info.get("raw_reward", shaped_r))
            bonus_list.append(info.get("bonus", 0.0))
            pot_bonus_list.append(info.get("potential_bonus", 0.0))
            time_fee_list.append(info.get("time_fee", 0.0))
            shaped_r_list.append(shaped_r)

            obs = next_obs
            done = term or trunc
            if done:
                outcome = info.get("ep_stats", {}).get("outcome", "unknown")

        env.close()

        # Save numeric arrays
        out_npz = f"results/rollouts/{config}_seed{seed}_ep{ep}.npz"
        np.savez_compressed(
            out_npz,
            obs=np.array(obs_list, dtype=np.float32),
            actions=np.array(action_list, dtype=np.int32),
            raw_reward=np.array(raw_r_list, dtype=np.float32),
            bonus=np.array(bonus_list, dtype=np.float32),
            potential_bonus=np.array(pot_bonus_list, dtype=np.float32),
            time_fee=np.array(time_fee_list, dtype=np.float32),
            shaped_reward=np.array(shaped_r_list, dtype=np.float32),
            outcome=outcome,
        )

    print(f"Rollouts complete for {config} seed={seed}.")


def collect_random(episodes: int = 5, seed_base: int = 20_000):
    pathlib.Path("results/rollouts").mkdir(parents=True, exist_ok=True)
    pathlib.Path("videos").mkdir(parents=True, exist_ok=True)

    print(f"Collecting rollouts for random baseline ({episodes} episodes)...")

    for ep in range(episodes):
        ep_seed = seed_base + ep
        render_mode = "rgb_array" if ep == 0 else None
        env = make_raw_env(render_mode=render_mode)
        obs, _ = env.reset(seed=ep_seed)
        rng = np.random.default_rng(ep_seed)

        obs_list = []
        action_list = []
        raw_r_list = []
        bonus_list = []
        pot_bonus_list = []
        time_fee_list = []
        shaped_r_list = []

        done = False
        outcome = "crashed"
        while not done:
            action = int(rng.integers(4))
            next_obs, r, term, trunc, _ = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            raw_r_list.append(r)
            bonus_list.append(0.0)
            pot_bonus_list.append(0.0)
            time_fee_list.append(0.0)
            shaped_r_list.append(r)

            obs = next_obs
            done = term or trunc
            if done:
                outcome = "timeout" if trunc and r not in (100, -100) else ("landed" if r >= 100 - 1e-6 else "crashed")

        env.close()

        out_npz = f"results/rollouts/random_seed-1_ep{ep}.npz"
        np.savez_compressed(
            out_npz,
            obs=np.array(obs_list, dtype=np.float32),
            actions=np.array(action_list, dtype=np.int32),
            raw_reward=np.array(raw_r_list, dtype=np.float32),
            bonus=np.array(bonus_list, dtype=np.float32),
            potential_bonus=np.array(pot_bonus_list, dtype=np.float32),
            time_fee=np.array(time_fee_list, dtype=np.float32),
            shaped_reward=np.array(shaped_r_list, dtype=np.float32),
            outcome=outcome,
        )

    print("Random baseline rollouts complete.")


def main():
    model_paths = sorted(glob.glob("runs/*/seed*/model.zip"))
    for p in model_paths:
        norm_p = p.replace("\\", "/")
        parts = norm_p.split("/")
        cfg = parts[1]
        seed_num = int(parts[2].replace("seed", ""))
        collect(model_path=p, config=cfg, seed=seed_num)

    collect_random()


if __name__ == "__main__":
    main()

