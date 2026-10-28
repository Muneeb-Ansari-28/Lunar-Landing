"""Main Interactive Console Launcher & Entry Point for Lunar Lander Simulation."""
from __future__ import annotations
import os
import sys

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import webbrowser
import pathlib
import subprocess
from ll.live_viewer import run_viewer


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print(r"""
================================================================================
             [#] LUNAR LANDER REINFORCEMENT LEARNING SIMULATOR [#]
================================================================================
  Real-Time Physics Simulation, Reward Shaping & Reward Hacking Experiments
================================================================================
""")


def print_menu():
    print("""  WATCH REAL-TIME LANDING VISUALS (Desktop Window):
    [1] Launch Live Landing: Trained Baseline Agent (Extended High Orbit)
    [2] Launch Live Simulation: Random Uncontrolled Rocket (Crash Test)
    [3] Launch Live Simulation: Hover Hack (Reward Hacked Agent)
    [4] Launch Live Simulation: Fixed Potential-Based Agent

  TRAINING & METRICS:
    [5] Train an AI Agent (Quick Smoke Test: ~45s)
    [6] Generate & Inspect Scientific Matplotlib Figures
    [7] Open Self-Contained HTML Report in Browser
    [8] Run Invariant Pytest Suite & System Check

    [0] Exit
================================================================================
  Controls in Live Simulation:
    [SPACE] Pause/Resume  |  [R] Restart  |  [A] High/Std Orbit
    [C] Cycle Policy      |  [1-4] Speed  |  [ESC/Q] Exit to Menu
================================================================================
""")


def main_menu():
    while True:
        clear_screen()
        print_banner()
        print_menu()
        try:
            choice = input("Enter option [0-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "1":
            print("\n>>> Launching Real-Time Baseline Landing Simulation (High Orbit)...")
            run_viewer(config="baseline", speed=1.0, altitude_mode="high")
            try:
                input("\nSimulation closed. Press [Enter] to return to menu...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == "2":
            print("\n>>> Launching Real-Time Random Rocket Crash Simulation...")
            run_viewer(config="random", speed=1.0, altitude_mode="high")
            try:
                input("\nSimulation closed. Press [Enter] to return to menu...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == "3":
            print("\n>>> Launching Real-Time Hover Hack Simulation...")
            run_viewer(config="hover_hack", speed=1.0, altitude_mode="high")
            try:
                input("\nSimulation closed. Press [Enter] to return to menu...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == "4":
            print("\n>>> Launching Real-Time Fixed Agent Landing Simulation...")
            run_viewer(config="fixed", speed=1.0, altitude_mode="high")
            try:
                input("\nSimulation closed. Press [Enter] to return to menu...")
            except (EOFError, KeyboardInterrupt):
                pass

        elif choice == "5":
            print("\nWhich configuration would you like to train?")
            print("  [1] baseline      (Learns standard safe landing)")
            print("  [2] hover_hack    (Learns hover hack exploit)")
            print("  [3] fixed         (Learns potential-based landing)")
            print("  [4] time_penalty  (Learns fast fuel-efficient landing)")
            sub_c = input("Select configuration [1-4] (default 1): ").strip()
            cfg_map = {"1": "baseline", "2": "hover_hack", "3": "fixed", "4": "time_penalty"}
            cfg = cfg_map.get(sub_c, "baseline")

            print("\nSelect training duration:")
            print("  [1] Quick Smoke Test   (100,000 steps ~45s)   - Basic posture learning")
            print("  [2] Standard Training  (500,000 steps ~3.5m)  - Safely lands on pad (Recommended)")
            print("  [3] Full Paper Target  (1,000,000 steps ~7m)  - High-precision expert landings")
            dur_choice = input("Select duration [1-3] (default 2): ").strip()

            step_map = {"1": 100_000, "2": 500_000, "3": 1_000_000}
            steps = step_map.get(dur_choice, 500_000)

            print(f"\n>>> Training {cfg.upper()} for {steps:,} timesteps with 16 parallel environments...")
            from ll.train import train
            train(config=cfg, seed=0, timesteps=steps, quick=False)
            input("\nTraining complete! Press [Enter] to return to menu...")

        elif choice == "6":
            print("\n>>> Generating matplotlib analysis figures...")
            from ll.plots import make_all
            make_all()
            print("Figures saved in figures/ folder.")
            input("\nPress [Enter] to return to menu...")

        elif choice == "7":
            print("\n>>> Opening HTML report...")
            from ll.report import build_report
            build_report()
            report_file = pathlib.Path("report/index.html").resolve()
            webbrowser.open(report_file.as_uri())
            input("\nReport opened. Press [Enter] to return to menu...")

        elif choice == "8":
            print("\n>>> Running pytest test suite...")
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
            input("\nTests finished. Press [Enter] to return to menu...")

        elif choice in ("0", "q", "quit", "exit"):
            print("\nExiting Lunar Lander Simulator. Safe travels!\n")
            break
        else:
            try:
                input("\nInvalid option. Press [Enter] to try again...")
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--config", default="baseline")
        ap.add_argument("--speed", type=float, default=1.0)
        a = ap.parse_args()
        run_viewer(config=a.config, speed=a.speed)
    else:
        main_menu()
