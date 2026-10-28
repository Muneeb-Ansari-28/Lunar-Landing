"""Phase 11: Checkpoint Evolution Media Generator.

Replays checkpoints saved across training milestones (10%, 25%, 50%, 100%)
under the identical evaluation seed on the raw environment.
"""
from __future__ import annotations
import glob
import pathlib
import imageio.v3 as iio
from stable_baselines3 import PPO
from ll.envs import make_raw_env
from ll.renderer import LunarRenderer


def generate_evolution_video(
    config: str = "baseline",
    seed: int = 0,
    eval_seed: int = 10_000,
    out_video: str = "videos/checkpoint_evolution.mp4",
):
    checkpoints_dir = pathlib.Path(f"runs/{config}/seed{seed}/checkpoints")
    cp_files = sorted(glob.glob(str(checkpoints_dir / "checkpoint_*_steps.zip")))
    final_model = pathlib.Path(f"runs/{config}/seed{seed}/model.zip")

    if final_model.exists():
        cp_files.append(str(final_model))

    if not cp_files:
        print(f"No checkpoints found for {config} seed {seed}.")
        return

    print(f"Found {len(cp_files)} checkpoints to evaluate for evolution video.")
    renderer = LunarRenderer(width=1280, height=720, headless=True)
    all_frames = []

    for idx, cp in enumerate(cp_files):
        print(f"Replaying milestone {idx + 1}/{len(cp_files)}: {cp}...")
        model = PPO.load(cp, device="cpu")
        env = make_raw_env()
        obs, _ = env.reset(seed=eval_seed)
        done = False
        frame_idx = 0
        raw_ret = 0.0

        while not done and frame_idx < 800:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, term, trunc, _ = env.step(action)
            raw_ret += reward
            frame_idx += 1
            done = term or trunc

            frame = renderer.render_frame(
                env=env,
                obs=obs,
                action=action,
                config=config,
                seed=seed,
                frame_idx=frame_idx,
                raw_ret=raw_ret,
                shaped_ret=raw_ret,
                outcome="landed" if (done and reward >= 99) else ("crashed" if (done and reward <= -99) else None),
            )
            all_frames.append(frame)

        env.close()

    renderer.close()

    if all_frames:
        pathlib.Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving evolution video to {out_video} ({len(all_frames)} frames)...")
        iio.imwrite(out_video, all_frames, fps=50)
        print("Evolution video complete.")


if __name__ == "__main__":
    generate_evolution_video()

