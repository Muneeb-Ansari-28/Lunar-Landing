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
    n_seeds = len({pathlib.Path(f).parent.name for f in files})
    suffix = " (Mean ± Std across Seeds)" if n_seeds > 1 else " (seed 0 only)"
    plt.title(f"Raw Episode Return vs. Training Timesteps{suffix}")
    plt.xlabel("Timesteps")
    plt.ylabel("Raw Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/01_learning_curves_raw.png")
    plt.close()


def fig03_episode_length():
    """Mean episode length vs. training timesteps, from progress.csv.

    A hovering policy stretches episodes toward the 1,000-frame timeout,
    so episode length is the direct signature of the hover exploit: if
    hover_hack were farming bonus, its episodes would lengthen while the
    other configs' shorten toward quick landings.
    """
    files = glob.glob("runs/*/seed*/progress.csv")
    if not files:
        print("Skipping Figure 03: No progress.csv files found.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    for cfg in ("baseline", "time_penalty", "hover_hack", "fixed"):
        cfg_files = glob.glob(f"runs/{cfg}/seed*/progress.csv")
        dfs = []
        for f in cfg_files:
            try:
                df = pd.read_csv(f)
                if len(df) > 5:
                    df["smooth_len"] = df["length"].rolling(100, min_periods=10).mean()
                    dfs.append(df)
            except Exception:
                continue

        if dfs:
            combined = pd.concat(dfs, ignore_index=True).sort_values("timestep")
            combined["step_bin"] = (combined["timestep"] // 5000) * 5000
            grouped = combined.groupby("step_bin")["smooth_len"].mean().reset_index()
            plt.plot(grouped["step_bin"], grouped["smooth_len"], label=cfg, color=COLORS.get(cfg, "blue"), lw=2)

    plt.axhline(1000, color="red", linestyle="--", alpha=0.7, label="Timeout (1,000 frames)")
    plt.title("Episode Length vs. Training Timesteps (Hovering = Long Episodes)")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Length (frames, rolling 100-ep)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/03_episode_length.png")
    plt.close()


def _load_rollouts():
    """Rollout observations keyed by config, from results/rollouts/*.npz.

    Returns {config: [(obs, outcome), ...]}; obs is (T, 8) with
    obs[0] = horizontal position, obs[1] = altitude.
    """
    data = {}
    for f in sorted(glob.glob("results/rollouts/*_seed*_ep*.npz")):
        cfg = pathlib.Path(f).name.split("_seed")[0]
        try:
            d = np.load(f)
            data.setdefault(cfg, []).append((d["obs"], str(d["outcome"])))
        except Exception:
            continue
    return data


def fig06_trajectories():
    """Flight paths (horizontal position vs. altitude) from rollout data.

    The landing pad is drawn from the env's own helipad coordinates
    (world units 8–12, i.e. obs x in [-0.2, 0.2], since obs x is
    (world_x - 10) / 10); defaults match the renderer's fallbacks.
    """
    data = _load_rollouts()
    if not data:
        print("Skipping Figure 06: no results/rollouts/*_seed*_ep*.npz found.")
        print("Run `python -m ll.rollouts` first to collect rollout data.")
        return

    try:
        from ll.envs import make_raw_env
        env = make_raw_env()
        env.reset(seed=0)
        u = env.unwrapped
        x1 = float(getattr(u, "helipad_x1", 8.0))
        x2 = float(getattr(u, "helipad_x2", 12.0))
        env.close()
        pad = ((x1 - 10.0) / 10.0, (x2 - 10.0) / 10.0)
    except Exception:
        pad = (-0.2, 0.2)

    plt.figure(figsize=(9, 5), dpi=160)
    for cfg in ("baseline", "time_penalty", "hover_hack", "fixed", "random"):
        eps = data.get(cfg)
        if not eps:
            continue
        color = COLORS.get(cfg, "blue")
        for obs, _outcome in eps:
            plt.plot(obs[:, 0], obs[:, 1], color=color, lw=0.9, alpha=0.4)
        plt.plot(eps[0][0][:, 0], eps[0][0][:, 1], color=color, lw=2,
                 label=f"{cfg} ({len(eps)} episodes)")

    plt.plot([pad[0], pad[1]], [0, 0], color="green", lw=5, solid_capstyle="butt",
             label="Landing pad")
    plt.axhline(0, color="black", lw=1)
    plt.title("Flight Trajectories: Horizontal Position vs. Altitude (Rollout Data)")
    plt.xlabel("Horizontal Position (obs x)")
    plt.ylabel("Altitude (obs y)")
    plt.ylim(-0.2, 1.6)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/06_trajectories.png")
    plt.close()


def fig07_altitude_vs_time():
    """Altitude over episode frames, from rollout data.

    A hovering policy holds altitude instead of descending: its curves
    stay high and flat, while a landing policy descends steadily to 0.
    Per-episode lines only — no mean is plotted, because episodes end at
    different frames and a mean over the episodes still running would
    jump whenever one ends.
    """
    data = _load_rollouts()
    if not data:
        print("Skipping Figure 07: no results/rollouts/*_seed*_ep*.npz found.")
        print("Run `python -m ll.rollouts` first to collect rollout data.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    for cfg in ("baseline", "time_penalty", "hover_hack", "fixed", "random"):
        eps = data.get(cfg)
        if not eps:
            continue
        color = COLORS.get(cfg, "blue")
        for obs, _outcome in eps:
            plt.plot(np.arange(len(obs)), obs[:, 1], color=color, lw=1.0, alpha=0.55)
        plt.plot([], [], color=color, lw=2, label=f"{cfg} ({len(eps)} episodes)")

    plt.title("Altitude vs. Time (Rollout Data)")
    plt.xlabel("Episode Frames")
    plt.ylabel("Altitude (obs y)")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/07_altitude_vs_time.png")
    plt.close()


def fig11_reward_components():
    """Cumulative raw vs. shaped reward per config, from rollout data.

    The gap between the two curves is the accumulated shaping bonus:
    for hover_hack it grows while the agent stays nice (the exploit),
    for fixed it telescopes back toward zero (the cure), and for
    baseline the curves coincide (no shaping). Per-episode lines only —
    no mean is plotted, because episodes end at different frames and a
    mean over the episodes still running would jump whenever one ends.
    """
    data = {}
    for f in sorted(glob.glob("results/rollouts/*_seed*_ep*.npz")):
        cfg = pathlib.Path(f).name.split("_seed")[0]
        try:
            d = np.load(f)
            data.setdefault(cfg, []).append((np.cumsum(d["raw_reward"]),
                                             np.cumsum(d["shaped_reward"])))
        except Exception:
            continue

    if not data:
        print("Skipping Figure 11: no results/rollouts/*_seed*_ep*.npz found.")
        print("Run `python -m ll.rollouts` first to collect rollout data.")
        return

    plt.figure(figsize=(9, 5), dpi=160)
    for cfg in ("baseline", "hover_hack", "fixed"):
        eps = data.get(cfg)
        if not eps:
            continue
        color = COLORS.get(cfg, "blue")
        for raw, shaped in eps:
            plt.plot(np.arange(len(raw)), raw, color=color, lw=1.0, alpha=0.55)
            plt.plot(np.arange(len(shaped)), shaped, color=color, lw=1.0, alpha=0.55,
                     linestyle="--")
        plt.plot([], [], color=color, lw=2, label=f"{cfg}: cumulative raw ({len(eps)} episodes)")
        plt.plot([], [], color=color, lw=2, linestyle="--",
                 label=f"{cfg}: cumulative shaped ({len(eps)} episodes)")

    plt.title("Reward Components: Cumulative Raw vs. Shaped Reward (Rollout Data)")
    plt.xlabel("Episode Frames")
    plt.ylabel("Cumulative Reward")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/11_reward_components.png")
    plt.close()


def fig13_seed_variance():
    """Mean eval return per config, with a spread bar.

    Reads results/eval_summary.csv (one row per config/seed). With two or
    more seeds per config the bar is the std across seeds — true seed
    variance. With only one seed (the current state) it falls back to the
    episode-level std from results/eval.csv and the title says so; run
    seeds 1–2 (item 15) for real seed variance.
    """
    p = pathlib.Path("results/eval_summary.csv")
    if not p.exists():
        print("Skipping Figure 13: results/eval_summary.csv not found.")
        print("Run `python -m ll.evaluate` first to produce evaluation data.")
        return

    df = pd.read_csv(p)
    grouped = df.groupby("config")["mean_return"].agg(["mean", "std", "count"])
    n_seeds = int(grouped["count"].max())
    if n_seeds >= 2:
        err = grouped["std"].fillna(0)
        note = f"error bars: std across {n_seeds} seeds"
    else:
        ep = pathlib.Path("results/eval.csv")
        if not ep.exists():
            print("Skipping Figure 13: results/eval.csv not found for episode spread.")
            return
        ep_df = pd.read_csv(ep)
        err = ep_df.groupby("config")["ret"].std().reindex(grouped.index).fillna(0)
        note = "single seed — error bars are episode spread, not seed variance"

    order = [c for c in ("baseline", "time_penalty", "hover_hack", "fixed", "random")
             if c in grouped.index]
    means = grouped.loc[order, "mean"]
    errs = err.loc[order]

    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(order, means, yerr=errs, capsize=6,
            color=[COLORS.get(c, "blue") for c in order], alpha=0.85)
    plt.axhline(200, color="green", linestyle="--", alpha=0.7, label="Solved (200)")
    plt.title(f"Mean Evaluation Return by Configuration\n({note})")
    plt.ylabel("Mean Return (50 episodes)")
    plt.xlabel("Configuration")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/13_seed_variance.png")
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
            plt.plot(df["timestep"], df["smooth_shaped"], color="red", lw=2, label="Shaped Return")
            plt.plot(df["timestep"], df["smooth_raw"], color="black", linestyle="--", lw=1.5, label="Raw Return")
            break

    # Title states what the 100k-step data actually shows: shaped tracks
    # raw with only a small bonus gap — the hover exploit has not emerged
    # yet (see audit §1). The original "rises while raw stalls" claim
    # described the intended phenomenon, not this smoke-test data.
    plt.title("Hover Hack: Shaped vs. Raw Return (100k Steps — No Exploit Yet)")
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
    fig03_episode_length()
    fig04_outcome_stack()
    fig05_eval_return_box()
    fig06_trajectories()
    fig07_altitude_vs_time()
    fig11_reward_components()
    fig12_fixed_bonus_bounded()
    fig13_seed_variance()
    print("Figures generation pass complete.")


if __name__ == "__main__":
    make_all()

