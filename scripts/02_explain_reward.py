"""Phase 2: Reproduce and verify the 2-frame reward table from the PDF paper."""
import pathlib
import gymnasium as gym
import numpy as np
import pandas as pd

FUEL = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}
ACTION_NAMES = {0: "noop", 1: "left_engine", 2: "main_engine", 3: "right_engine"}

def calc_shaping(obs):
    x, y, vx, vy, angle, _, left_leg, right_leg = obs
    dist = np.sqrt(x**2 + y**2)
    speed = np.sqrt(vx**2 + vy**2)
    tilt = abs(angle)
    legs = 10.0 * (left_leg + right_leg)
    pos_score = -100.0 * dist - 100.0 * speed - 100.0 * tilt + legs
    return dist, speed, tilt, legs, pos_score

def main():
    pathlib.Path("results").mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3")
    obs, info = env.reset(seed=42)

    rows = []
    dist, speed, tilt, legs, prev_shaping = calc_shaping(obs)
    print("=== Reward Explainer (Matching Research Paper) ===")
    print(f"Initial State: dist={dist:.3f}, speed={speed:.3f}, tilt={tilt:.3f}, shaping={prev_shaping:.3f}\n")

    for frame in range(1, 11):
        action = env.action_space.sample()
        fuel_cost = FUEL[action]
        obs, reward, terminated, truncated, _ = env.step(action)
        dist, speed, tilt, legs, current_shaping = calc_shaping(obs)
        delta_shaping = current_shaping - prev_shaping
        expected_reward = delta_shaping - fuel_cost

        if not (terminated or truncated):
            diff = abs(reward - expected_reward)
            assert diff < 1e-2, f"Frame {frame}: reward {reward:.4f} != expected {expected_reward:.4f} (diff={diff})"

        rows.append({
            "frame": frame,
            "action": ACTION_NAMES[action],
            "distance": round(dist, 4),
            "speed": round(speed, 4),
            "tilt": round(tilt, 4),
            "legs_down": int((legs > 0)),
            "position_score": round(current_shaping, 4),
            "delta_shaping": round(delta_shaping, 4),
            "fuel_cost": round(fuel_cost, 4),
            "actual_reward": round(reward, 4),
            "expected_reward": round(expected_reward, 4),
        })

        prev_shaping = current_shaping
        if terminated or truncated:
            break

    env.close()

    df = pd.DataFrame(rows)
    csv_path = "results/reward_explainer.csv"
    df.to_csv(csv_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved reward explanation table to {csv_path}")
    print("Verification passed: Actual rewards match theoretical delta shaping minus fuel costs.")

if __name__ == "__main__":
    main()

