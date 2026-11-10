"""Phase 8: Scientific Publication Figures using Matplotlib."""
from __future__ import annotations
import glob
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "random": "#808080",
    "baseline": "#1f77b4",
    "time_penalty": "#17becf",
    "hover_hack": "#d62728",
    "fixed": "#2ca02c",
}


def fig01_learning_curves_raw():
    files = glob.glob("runs/*/seed*/progress.csv")
    if not files:
        print("Skipping Figure 01: No progress.csv files found.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    configs = ["baseline", "time_penalty", "hover_hack", "fixed"]

    for cfg in configs:
        cfg_files = glob.glob(f"runs/{cfg}/seed*/progress.csv")
        all_dfs = []
        for f in cfg_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 5:
                    df["smooth_ret"] = df["raw_return"].rolling(100, min_periods=10).mean()
                    all_dfs.append(df)
            except Exception:
                continue

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True).sort_values("timestep")
            combined["step_bin"] = (combined["timestep"] // 5000) * 5000
            grouped = combined.groupby("step_bin")["smooth_ret"].agg(["mean", "std"]).reset_index()
            plt.plot(grouped["step_bin"], grouped["mean"], label=cfg, color=COLORS.get(cfg, "blue"), lw=2)
            plt.fill_between(
                grouped["step_bin"],
                grouped["mean"] - grouped["std"].fillna(0),
                grouped["mean"] + grouped["std"].fillna(0),
                alpha=0.15,
                color=COLORS.get(cfg, "blue"),
            )

    plt.axhline(200, color="green", linestyle="--", alpha=0.7, label="Solved (Return >= 200)")
    plt.title("Raw Episode Return vs. Training Timesteps (Mean ± Std across Seeds)")
    plt.xlabel("Timesteps")
    plt.ylabel("Raw Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/01_learning_curves_raw.png")
    plt.close()


def fig02_raw_vs_shaped_hack():
    files = glob.glob("runs/hover_hack/seed*/progress.csv")
    if not files:
        print("Skipping Figure 02: No hover_hack progress files found.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    for f in files:
        df = pd.read_csv(f)
        if len(df) > 10:
            df["smooth_raw"] = df["raw_return"].rolling(100, min_periods=10).mean()
            df["smooth_shaped"] = df["shaped_return"].rolling(100, min_periods=10).mean()
            plt.plot(df["timestep"], df["smooth_shaped"], color="red", lw=2, label="Shaped Return (Rising)")
            plt.plot(df["timestep"], df["smooth_raw"], color="black", linestyle="--", lw=1.5, label="Raw Return (Stalled)")
            break

    plt.title("Reward Hacking: Shaped Return Rises While Raw Return Stalls")
    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/02_raw_vs_shaped_hack.png")
    plt.close()


def fig04_outcome_stack():
    p = pathlib.Path("results/eval.csv")
    if not p.exists():
        print("Skipping Figure 04: results/eval.csv not found.")
        return

    df = pd.read_csv(p)
    outcomes = df.groupby(["config", "outcome"]).size().unstack(fill_value=0)
    outcomes = outcomes.div(outcomes.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(9, 5), dpi=160)
    outcomes.plot(kind="bar", stacked=True, color={"landed": "#2ca02c", "crashed": "#d62728", "timeout": "#ff7f0e"}, ax=plt.gca())
    plt.title("Episode Outcomes by Configuration (50 Eval Episodes per Seed)")
    plt.ylabel("Percentage (%)")
    plt.xlabel("Configuration")
    plt.legend(title="Outcome")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/04_outcome_stack.png")
    plt.close()


def fig05_eval_return_box():
    p = pathlib.Path("results/eval.csv")
    if not p.exists():
        print("Skipping Figure 05: results/eval.csv not found.")
        return

    df = pd.read_csv(p)
    plt.figure(figsize=(9, 5), dpi=160)
    df.boxplot(column="ret", by="config", grid=False, patch_artist=True)
    plt.suptitle("")
    plt.title("Evaluation Return Distribution (Raw Environment, 50 Episodes)")
    plt.axhline(200, color="green", linestyle="--", label="Solved (200)")
    plt.ylabel("Return")
    plt.xlabel("Configuration")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/05_eval_return_box.png")
    plt.close()


def fig12_fixed_bonus_bounded():
    """Cumulative shaping bonus per frame, measured from rollout .npz files.

    For each of hover_hack and fixed, plots every collected episode's
    cumulative `bonus` (thin lines) plus the mean across episodes (thick
    line). The dashed line is the analytic bound for `fixed`, derived from
    ll/envs.py: the niceness potential Φ lies in [0, 2], so the telescoped
    potential sum over an episode is at most γ·2, and the cumulative bonus
    (potential sum minus the per-frame time fee) is at most γ·2 − 0.05·frame.
    If no rollout data exists, the figure is skipped instead of drawn from
    invented curves — run `python -m ll.rollouts` first.
    """
    p = pathlib.Path("figures")
    p.mkdir(parents=True, exist_ok=True)

    data = {}
    for cfg in ("hover_hack", "fixed"):
        curves = []
        for f in sorted(glob.glob(f"results/rollouts/{cfg}_seed*_ep*.npz")):
            d = np.load(f)
            curves.append(np.cumsum(d["bonus"]))
        if curves:
            data[cfg] = curves

    if not data:
        print("Skipping Figure 12: no results/rollouts/{hover_hack,fixed}_seed*_ep*.npz found.")
        print("Run `python -m ll.rollouts` first to collect rollout data.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    max_len = 0
    for cfg, curves in data.items():
        color = COLORS.get(cfg, "blue")
        for c in curves:
            plt.plot(np.arange(len(c)), c, color=color, lw=0.8, alpha=0.35)
        # Mean across episodes still running at each frame.
        n = max(len(c) for c in curves)
        mean = np.array([np.mean([c[t] for c in curves if len(c) > t]) for t in range(n)])
        plt.plot(np.arange(n), mean, color=color, lw=2.5,
                 label=f"{cfg}: mean cumulative bonus ({len(curves)} episodes)")
        max_len = max(max_len, n)

    # Analytic upper bound for `fixed` (envs.py defaults: gamma=0.999,
    # time_fee=0.05, niceness potential Φ in [0, 2]).
    frames = np.arange(0, max_len)
    plt.plot(frames, 0.999 * 2.0 - 0.05 * frames, color="black", linestyle="--", lw=1.5,
             label="fixed: theoretical bound (γ·2 − 0.05·frame)")

    plt.title("Measured Cumulative Shaping Bonus per Episode (Rollout Data)")
    plt.xlabel("Episode Frames")
    plt.ylabel("Cumulative Bonus Received")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/12_fixed_bonus_bounded.png")
    plt.close()
    print(f"Figure 12: plotted { {cfg: len(c) for cfg, c in data.items()} } from rollout data.")


def make_all():
    pathlib.Path("figures").mkdir(parents=True, exist_ok=True)
    print("Generating figures...")
    fig01_learning_curves_raw()
    fig02_raw_vs_shaped_hack()
    fig04_outcome_stack()
    fig05_eval_return_box()
    fig12_fixed_bonus_bounded()
    print("Figures generation pass complete.")


if __name__ == "__main__":
    make_all()

