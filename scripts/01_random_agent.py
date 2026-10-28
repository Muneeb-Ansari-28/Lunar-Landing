"""Phase 2: Random Agent Baseline & Crash Video Generator."""
import pathlib
import gymnasium as gym
import imageio.v3 as iio
import numpy as np

OBS_LABELS = [
    "x (left/right of pad)",
    "y (height above pad)",
    "vx (sideways speed)",
    "vy (vertical speed)",
    "angle (tilt in radians)",
    "angular_velocity (spin)",
    "left_leg_contact (0/1)",
    "right_leg_contact (0/1)",
]

def main():
    pathlib.Path("videos").mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    
    print("=== Running 5 Episodes of Random Policy ===")
    all_frames = []

    for ep in range(5):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        steps = 0
        frames = []
        terminated = False
        truncated = False

        if ep == 0:
            print("\nObservation readout for first 3 frames of Episode 0:")

        while not (terminated or truncated):
            if ep == 0:
                frames.append(env.render())
                if steps < 3:
                    print(f"\n--- Frame {steps} ---")
                    for val, label in zip(obs, OBS_LABELS):
                        print(f"  {label:<26}: {val:+.4f}")

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {ep + 1}: Return = {total_reward:.2f}, Length = {steps} frames")

        if ep == 0:
            all_frames = frames

    env.close()

    if all_frames:
        mp4_path = "videos/random_crash.mp4"
        gif_path = "videos/random_crash.gif"
        print(f"\nWriting {mp4_path} ({len(all_frames)} frames @ 50 fps)...")
        iio.imwrite(mp4_path, all_frames, fps=50)

        # For GIF, downsample to 25 fps (stride 2) for reasonable size
        gif_frames = all_frames[::2]
        print(f"Writing {gif_path} ({len(gif_frames)} frames @ 25 fps)...")
        iio.imwrite(gif_path, gif_frames, fps=25, loop=0)
        print("Videos saved successfully.")

if __name__ == "__main__":
    main()

