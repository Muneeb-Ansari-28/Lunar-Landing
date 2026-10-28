"""Phase 4: PPO Training Engine with Per-Episode Metrics and Checkpointing."""
from __future__ import annotations
import argparse
import pathlib
import time
import torch
from functools import partial
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from ll.envs import ENV_ID, ShapedReward


class EpisodeLogger(BaseCallback):
    """Writes one CSV row per finished episode (raw AND shaped return)."""

    def __init__(self, path: pathlib.Path | str):
        super().__init__()
        self.path = str(path)
        self.f = None

    def _on_training_start(self) -> None:
        self.f = open(self.path, "w", encoding="utf-8")
        self.f.write("timestep,raw_return,shaped_return,length,outcome\n")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            s = info.get("ep_stats")
            if s:
                self.f.write(
                    f"{self.num_timesteps},{s['raw_return']:.3f},{s['shaped_return']:.3f},"
                    f"{s['length']},{s['outcome']}\n"
                )
        return True

    def _on_rollout_end(self) -> None:
        if self.f:
            self.f.flush()

    def _on_training_end(self) -> None:
        if self.f:
            self.f.close()


def train(config: str, seed: int = 0, timesteps: int = 1_000_000, n_envs: int = 16, quick: bool = False):
    torch.set_num_threads(1)  # one thread per process so parallel seeds don't oversubscribe cores
    steps = 100_000 if quick else timesteps
    mode = {"baseline": "none"}.get(config, config)
    out = pathlib.Path(f"runs/{config}/seed{seed}")
    out.mkdir(parents=True, exist_ok=True)

    print(f"[{config} seed={seed}] Starting training: {steps} steps, n_envs={n_envs}, mode='{mode}'...")
    start_time = time.time()

    # NOTE: make_vec_env applies SB3's Monitor FIRST (inner), then wrapper_class,
    # so SB3's ep_rew_mean in TensorBoard is the RAW return. Shaped return is in progress.csv.
    env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=seed,
        wrapper_class=partial(ShapedReward, mode=mode),
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=seed,
        device="cpu",
        verbose=1,
        tensorboard_log=str(out / "tb"),
    )

    save_freq = max(10_000, (steps // n_envs) // 10)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(out / "checkpoints"),
        name_prefix="checkpoint",
    )
    episode_logger = EpisodeLogger(out / "progress.csv")

    model.learn(total_timesteps=steps, callback=[episode_logger, checkpoint_callback])
    model.save(str(out / "model"))

    elapsed = time.time() - start_time
    print(f"[{config} seed={seed}] Finished in {elapsed:.1f}s. Model saved to {out / 'model.zip'}")

    timing_file = pathlib.Path("results/timing.csv")
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not timing_file.exists()
    with open(timing_file, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("config,seed,timesteps,elapsed_sec\n")
        f.write(f"{config},{seed},{steps},{elapsed:.2f}\n")


def main():
    ap = argparse.ArgumentParser(description="Train PPO on LunarLander-v3 with shaped reward.")
    ap.add_argument("--config", required=True, choices=["baseline", "time_penalty", "hover_hack", "fixed"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="100k steps smoke test")
    a = ap.parse_args()

    train(config=a.config, seed=a.seed, timesteps=a.timesteps, n_envs=a.n_envs, quick=a.quick)


if __name__ == "__main__":
    main()

