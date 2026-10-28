import gymnasium as gym
import stable_baselines3
import torch
import imageio
import matplotlib

def check():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, info = env.reset(seed=0)
    frame = env.render()
    print("=== Installation Verification ===")
    print(f"gymnasium:         {gym.__version__}")
    print(f"stable-baselines3: {stable_baselines3.__version__}")
    print(f"torch:             {torch.__version__}")
    print(f"imageio:           {imageio.__version__}")
    print(f"matplotlib:        {matplotlib.__version__}")
    print(f"obs shape:         {obs.shape}")
    print(f"frame shape:       {frame.shape}")
    print("=================================")
    assert obs.shape == (8,), f"Expected obs shape (8,), got {obs.shape}"
    assert frame.ndim == 3 and frame.shape[2] == 3, f"Expected RGB frame, got {frame.shape}"
    print("SUCCESS: Headless LunarLander-v3 environment initialized and rendered successfully!")
    env.close()

if __name__ == "__main__":
    check()

