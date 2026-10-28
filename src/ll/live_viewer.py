"""Phase 6: Interactive Desktop Live Viewer for Lunar Lander Inference."""
from __future__ import annotations
import argparse
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import time
import imageio.v3 as iio
import numpy as np
import pygame
from stable_baselines3 import PPO
from ll.envs import make_env, make_raw_env
from ll.renderer import LunarRenderer

CONFIG_ORDER = ["baseline", "hover_hack", "fixed", "random"]


def run_viewer(
    config: str = "baseline",
    seed: int = 0,
    model_path: str | None = None,
    episodes: int = 99999,
    speed: float = 1.0,
    env_mode: str = "raw",
    headless: bool = False,
    record_path: str | None = None,
    deterministic: bool = True,
    altitude_mode: str = "high",
):
    current_config = config
    loaded_models = {}

    def get_model(cfg: str):
        if cfg == "random":
            return None
        if cfg in loaded_models:
            return loaded_models[cfg]
        p = pathlib.Path(f"runs/{cfg}/seed{seed}/model.zip")
        if p.exists():
            print(f">>> Loading model for {cfg} from {p}...")
            m = PPO.load(str(p), device="cpu")
            loaded_models[cfg] = m
            return m
        else:
            print(f"\n[INFO] Model checkpoint for '{cfg}' not found at {p}.")
            print("Falling back to random actions until model is trained.")
            return None

    model = get_model(current_config)

    renderer = LunarRenderer(width=1280, height=720, headless=headless)
    recorded_frames = []

    clock = pygame.time.Clock()
    target_fps = int(50 * speed)

    paused = False
    step_once = False
    running = True

    print(f"\n=======================================================")
    print(f"🚀 LUNAR LANDER REAL-TIME DESKTOP SIMULATION STARTED")
    print(f"=======================================================")
    print(f"  Active Policy: {current_config.upper()} (Seed {seed})")
    print(f"  Altitude:      {altitude_mode.upper()} ORBIT DESCENT (Extended Vertical Expanse)")
    print(f"  Mode:          {env_mode.upper()} environment")
    print(f"  Speed:         {speed}x (50 FPS physics)")
    print(f"-------------------------------------------------------")
    print("  Controls:")
    print("    [SPACE]   Pause / Resume simulation")
    print("    [R]       Restart / Reset current landing attempt")
    print("    [A]       Toggle Altitude Profile (High Orbit vs Standard)")
    print("    [C]       Cycle policy (Baseline -> Hover -> Fixed -> Random)")
    print("    [1-4]     Set playback speed (0.5x, 1x, 2x, 4x)")
    print("    [ESC / Q] Quit viewer & return to menu")
    print(f"=======================================================\n")

    ep = 0
    while ep < episodes and running:
        # Create fresh environment
        if env_mode == "wrapped":
            env = make_env(mode={"baseline": "none"}.get(current_config, current_config))
        else:
            env = make_raw_env()

        obs, _ = env.reset(seed=20_000 + ep)
        if altitude_mode == "high":
            dy = 5.8
            env.unwrapped.lander.position = (
                env.unwrapped.lander.position[0],
                env.unwrapped.lander.position[1] + dy,
            )
            for leg in env.unwrapped.legs:
                leg.position = (leg.position[0], leg.position[1] + dy)
            obs[1] += dy / 6.66666667

        raw_ret = 0.0
        shaped_ret = 0.0
        frame_idx = 0
        done = False
        outcome = None

        while not done and running:
            # Event handling
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print(">>> Simulation", "PAUSED" if paused else "RESUMED")
                        elif event.key == pygame.K_s:
                            step_once = True
                        elif event.key == pygame.K_r:
                            done = True  # reset episode
                            print(">>> Resetting attempt...")
                        elif event.key == pygame.K_a:
                            altitude_mode = "standard" if altitude_mode == "high" else "high"
                            print(f">>> Switched altitude profile to: {altitude_mode.upper()} ORBIT (restarting attempt...)")
                            done = True
                        elif event.key == pygame.K_c:
                            # Cycle policy
                            curr_idx = CONFIG_ORDER.index(current_config) if current_config in CONFIG_ORDER else 0
                            current_config = CONFIG_ORDER[(curr_idx + 1) % len(CONFIG_ORDER)]
                            model = get_model(current_config)
                            print(f">>> Switched policy to: {current_config.upper()}")
                        elif event.key == pygame.K_1:
                            target_fps = 25
                            speed = 0.5
                        elif event.key == pygame.K_2:
                            target_fps = 50
                            speed = 1.0
                        elif event.key == pygame.K_3:
                            target_fps = 100
                            speed = 2.0
                        elif event.key == pygame.K_4:
                            target_fps = 200
                            speed = 4.0

            if not paused or step_once:
                step_once = False

                if model:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    action = int(action)
                else:
                    action = int(env.action_space.sample())

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if "raw_reward" in info:
                    raw_ret += info["raw_reward"]
                    shaped_ret += reward
                else:
                    raw_ret += reward
                    shaped_ret += reward

                frame_idx += 1

                if done:
                    if truncated and not terminated:
                        outcome = "timeout"
                    elif reward >= 100 - 1e-6:
                        outcome = "landed"
                    else:
                        outcome = "crashed"

            # Render scene
            frame = renderer.render_frame(
                env=env,
                obs=obs,
                action=action if (not paused or step_once) else 0,
                config=current_config,
                seed=seed,
                frame_idx=frame_idx,
                raw_ret=raw_ret,
                shaped_ret=shaped_ret,
                outcome=outcome,
                custom_hud_data={"altitude_mode": altitude_mode},
                paused=paused,
                speed=speed,
            )

            if record_path:
                recorded_frames.append(frame)

            if not headless:
                clock.tick(target_fps)

        print(f"Attempt {ep + 1}: Outcome = {outcome.upper() if outcome else 'STOPPED'}, Return = {raw_ret:.1f}, Steps = {frame_idx}")

        # Post-episode brief pause so user can see result banner before resetting
        if not headless and running and done:
            for _ in range(50):  # ~1 second pause
                if not running:
                    break
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE)):
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        running = True
                        break
                renderer.render_frame(
                    env=env,
                    obs=obs,
                    action=0,
                    config=current_config,
                    seed=seed,
                    frame_idx=frame_idx,
                    raw_ret=raw_ret,
                    shaped_ret=shaped_ret,
                    outcome=outcome,
                    custom_hud_data={"altitude_mode": altitude_mode},
                    paused=paused,
                    speed=speed,
                )
                clock.tick(50)

        env.close()
        ep += 1

    renderer.close()

    if record_path and recorded_frames:
        pathlib.Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving recorded video to {record_path} ({len(recorded_frames)} frames)...")
        iio.imwrite(record_path, recorded_frames, fps=50)
        print("Saved successfully.")


def main():
    ap = argparse.ArgumentParser(description="Live Desktop Viewer for Lunar Lander.")
    ap.add_argument("--config", default="baseline", choices=["baseline", "time_penalty", "hover_hack", "fixed", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=99999)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--env", choices=["raw", "wrapped"], default="raw", dest="env_mode")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", type=str, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--altitude", choices=["high", "standard"], default="high")
    a = ap.parse_args()

    run_viewer(
        config=a.config,
        seed=a.seed,
        model_path=a.model,
        episodes=a.episodes,
        speed=a.speed,
        env_mode=a.env_mode,
        headless=a.headless,
        record_path=a.record,
        deterministic=not a.stochastic,
        altitude_mode=a.altitude,
    )


if __name__ == "__main__":
    main()
