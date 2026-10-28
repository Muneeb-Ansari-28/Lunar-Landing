"""Phase 5: Evaluation CLI on Raw Environment (50 Episodes, Deterministic)."""
from __future__ import annotations
import glob
import pathlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from ll.envs import make_raw_env

N_EPISODES = 50
FUEL = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}


def eval_model(path: str | pathlib.Path, n: int = N_EPISODES) -> pd.DataFrame:
    model = PPO.load(str(path), device="cpu")
    env = make_raw_env()
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        done, ret, steps, fuel, last_r, trunc = False, 0.0, 0, 0.0, 0.0, False
        actions = np.zeros(4, int)
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            a = int(a)
            actions[a] += 1
            fuel += FUEL[a]
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            steps += 1
            last_r = r
            done = term or trunc

        outcome = "timeout" if (trunc and last_r not in (100, -100)) else ("landed" if last_r >= 100 - 1e-6 else "crashed")
        row = dict(
            ret=ret,
            length=steps,
            outcome=outcome,
            final_x=float(obs[0]),
            fuel=fuel,
            **{f"act{k}": int(actions[k]) for k in range(4)},
        )
        rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def eval_random(n: int = N_EPISODES) -> pd.DataFrame:
    """Same 50-episode protocol with a seeded random policy (no model file)."""
    env = make_raw_env()
    rows = []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        rng = np.random.default_rng(10_000 + i)
        done, ret, steps, fuel, last_r, trunc = False, 0.0, 0, 0.0, 0.0, False
        actions = np.zeros(4, int)
        while not done:
            a = int(rng.integers(4))
            actions[a] += 1
            fuel += FUEL[a]
            obs, r, term, trunc, _ = env.step(a)
            ret += r
            steps += 1
            last_r = r
            done = term or trunc

        outcome = "timeout" if (trunc and last_r not in (100, -100)) else ("landed" if last_r >= 100 - 1e-6 else "crashed")
        row = dict(
            ret=ret,
            length=steps,
            outcome=outcome,
            final_x=float(obs[0]),
            fuel=fuel,
            **{f"act{k}": int(actions[k]) for k in range(4)},
        )
        rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def main():
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    all_rows = []

    model_paths = sorted(glob.glob("runs/*/seed*/model.zip"))
    print(f"Found {len(model_paths)} trained models to evaluate.")

    for p in model_paths:
        norm_p = p.replace("\\", "/")
        parts = norm_p.split("/")
        cfg = parts[1]
        seed_str = parts[2]
        seed_num = int(seed_str.replace("seed", ""))
        print(f"Evaluating {cfg} (seed {seed_num})...")
        df = eval_model(p)
        df["config"] = cfg
        df["seed"] = seed_num
        all_rows.append(df)

    print("Evaluating seeded random baseline...")
    rand_df = eval_random()
    rand_df["config"] = "random"
    rand_df["seed"] = -1
    all_rows.append(rand_df)

    full_df = pd.concat(all_rows, ignore_index=True)
    eval_csv = "results/eval.csv"
    full_df.to_csv(eval_csv, index=False)
    print(f"Saved full evaluation metrics to {eval_csv}")

    summary = (
        full_df.groupby(["config", "seed"])
        .agg(
            mean_return=("ret", "mean"),
            std_return=("ret", "std"),
            mean_len=("length", "mean"),
            landed=("outcome", lambda s: float((s == "landed").mean())),
            timeout=("outcome", lambda s: float((s == "timeout").mean())),
            crashed=("outcome", lambda s: float((s == "crashed").mean())),
        )
        .reset_index()
    )
    summary_csv = "results/eval_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nSaved evaluation summary to {summary_csv}:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

