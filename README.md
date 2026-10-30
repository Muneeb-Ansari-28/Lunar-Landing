# 🚀 Lunar Lander: Reinforcement Learning, Reward Design & Live Simulation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-orange.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-v2.0%2B-red.svg)](https://stable-baselines3.readthedocs.io/)
[![Pygame-CE](https://img.shields.io/badge/Pygame--CE-v2.5%2B-brightgreen.svg)](https://pyga.me/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

This end-to-end Reinforcement Learning project examines **reward shaping** and **reward hacking** in Gymnasium's Box2D `LunarLander-v3`. It includes a custom high-definition aerospace simulator, pre-trained PPO agents, and an interactive desktop flight console.

Reproducible experiments and live visualizations work together here, making it possible to study reward-design trade-offs through both metrics and flight behavior.

---

## 📸 Simulation Showcase

### 1. High-Altitude Orbital Descent

The spacecraft begins from high orbit ($y \approx 20\,\text{m}$), uses retrograde burns to reduce its velocity, and corrects horizontal drift against an expansive cosmic backdrop:
![High Altitude Orbital Descent](figures/high_orbit_snap.png)

### 2. Precision Touchdown at Pad `LZ-1 ALPHA`

The agent performs an autonomous deceleration burn and precise touchdown while the simulator displays locked landing gear, approach-runway LEDs, and flight telemetry:
![Touchdown](figures/live_snap_touchdown.png)

---

## 🧠 Reward Design and the "Hover Hack"

An RL agent optimizes the mathematical signal it receives, which can lead to clever but unintended strategies:

1. **The Core Task:**
   The agent receives an 8-dimensional state vector containing position, velocity, angle, angular velocity, and leg-contact signals. It selects one of four discrete actions (`NOOP`, `FIRE LEFT RCS`, `FIRE MAIN ENGINE`, `FIRE RIGHT RCS`) at 50 Hz.

2. **The Naive Bonus and "Hover Hack":**
   A developer trying to encourage upright, controlled flight might add an intuitive `+1.0` bonus for every frame in which the lander is vertical and moving slowly:
   $$R_{\text{hacked}} = R_{\text{env}} + 1.0 \quad \text{if } |x| < 0.2 \text{ and } |\theta| < 0.1$$
   **The result:** Landing ends the episode and stops the bonus. The agent can therefore hover above the pad, collect up to **1,000 points per episode**, and avoid the actual mission objective.

3. **The Solution: Potential-Based Reward Shaping (PBRS):**
   To avoid distorting the policy, the project implements Ng, Harada, and Russell's (1999) potential-based reward-shaping theorem:
   $$F(s, s') = \gamma \Phi(s') - \Phi(s)$$
   with $\Phi(s_{\text{terminal}}) = 0$, combined with an explicit per-step time penalty ($-0.05$). Over any trajectory, the shaped reward component telescopes:
   $$\sum_{t=0}^{T} F(s_t, s_{t+1}) = -\Phi(s_0) \le 0$$
   This telescoping property prevents unbounded reward farming while preserving policy invariance.

| Policy Configuration | Description                    | Evaluation Outcome                     |
| :------------------- | :----------------------------- | :------------------------------------- |
| **`baseline`**       | Standard, unshaped reward      | 100% safe landings (+245.3 average return) |
| **`hover_hack`**     | Upright-bonus exploit          | Hovers indefinitely and never lands     |
| **`fixed`**          | Potential-based shaping (PBRS) | Fast, fuel-efficient safe touchdown     |
| **`random`**         | Untrained stochastic policy    | Immediate crash or tumble               |

---

## 🎮 Interactive Desktop Flight Simulator

The custom Pygame-CE renderer provides:

- **Dual-layer additive supersonic plume:** Mach shock diamonds and turbulent plasma tendrils rendered with `BLEND_RGBA_ADD` and dynamic flicker.
- **Accurate nozzle alignment:** The exhaust root begins inside the engine bell without a detached air gap.
- **Cold-gas RCS thrusters:** Directional hypergolic/nitrogen vapor jets emit from the cabin quads.
- **Photorealistic lunar regolith:** A high-resolution surface uses Lambertian directional lighting derived from the sun vector.
- **High-tech landing platform:** The platform includes a machined carbon-titanium foundation, hazard chevrons, crosshair reticle, runway LEDs, and strobe beacons.
- **Dynamic ground lighting and dust ejecta:** Engine spotlights illuminate the regolith while horizontal dust rays respond to flight.
- **Glass-cockpit telemetry HUD:** The HUD shows altitude, sink-rate safety, pitch attitude, contact sensors, and score in real time.

### Simulator Hotkeys:

|       Key       | Action                                                                             |
| :-------------: | :--------------------------------------------------------------------------------- |
|  **`[SPACE]`**  | Pause or resume the real-time simulation                                          |
|    **`[R]`**    | Restart or reset the current landing attempt                                      |
|    **`[A]`**    | **Toggle altitude profile** (high orbit $20\,\text{m}$ or standard $13\,\text{m}$) |
|    **`[C]`**    | **Cycle policy** (`baseline` $\to$ `hover_hack` $\to$ `fixed` $\to$ `random`)     |
|   **`[1-4]`**   | Set the warp-speed factor ($0.5\times$, $1.0\times$, $2.0\times$, $4.0\times$)    |
| **`[ESC / Q]`** | Exit the viewer and return to the launcher menu                                   |

---

## ⚡ Quick Start

### 1. Clone the project

```bash
git clone https://github.com/Muneeb-Ansari-28/Lunar-Landing.git
cd Lunar-Landing
```

### 2. Install the dependencies

```bash
# Python 3.10 or newer is recommended
pip install -r requirements.txt
pip install -e .
```

### 3. Start the interactive console

On Windows:

```powershell
.\run.ps1
```

Or directly with Python:

```bash
python main.py
```

### Useful CLI commands

```bash
# Start a high-altitude orbital descent:
python -m ll.live_viewer --config baseline --altitude high

# Observe the hover-hack policy:
python -m ll.live_viewer --config hover_hack

# Run a random-policy crash test:
python -m ll.live_viewer --config random

# Run the unit and property tests:
pytest tests/ -v
```

---

## 📂 Repository Structure

```
Lunar_Landing/
├── main.py                     # Main interactive launcher console
├── run.ps1                     # PowerShell automation script for Windows
├── pyproject.toml              # Build metadata and editable install definition
├── requirements.txt            # Pinned dependencies
├── assets/                     # High-res sprites, textures, and backdrops
├── src/ll/
│   ├── envs.py                 # ShapedReward wrapper & environment factories
│   ├── train.py                # PPO training pipeline with EpisodeLogger
│   ├── evaluate.py             # 50-episode raw-environment evaluator
│   ├── renderer.py             # High-definition Pygame-CE renderer & HUD
│   ├── live_viewer.py          # Interactive desktop viewer CLI
│   ├── ui_theme.py             # UI color palettes, typography & layout
│   ├── rollouts.py             # Trajectory collector
│   ├── video.py                # Video & GIF streaming encoders
│   ├── plots.py                # Scientific Matplotlib figure generator
│   └── report.py               # Standalone HTML report generator
├── tests/
│   ├── test_envs.py            # Unit & property tests for reward shaping
│   ├── test_renderer.py        # Presentation coordinate mapping tests
│   └── test_live_viewer.py     # Live viewer headless execution tests
├── runs/                       # Pre-trained policy weights (model.zip)
├── figures/                    # Simulation and analytical figures
├── videos/                     # Demo MP4 recordings
└── report/                     # Self-contained evaluation report
```

---

## 🧪 Verification and Invariant Testing

The test suite includes unit and invariant property-based checks for algebraic bounds and environment stability:

```bash
pytest tests/ -v
```

The seven invariant checks cover:

- Bounded telescoping sums $\sum F(s, s') \le 0$
- Safe-touchdown detection versus timeout truncation
- Headless rendering output shape `(720, 1280, 3)`
- Coordinate transformation mapping $(0, 0) \to (0, 720)$ and $(W, H) \to (1280, 0)$

---

## 📜 References

- Ng, A. Y., Harada, D., & Russell, S. (1999). _Policy invariance under reward transformations: Theory and application to reward shaping_. ICML.
- Farama Foundation Gymnasium: [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
- Stable-Baselines3: Reliable Reinforcement Learning Implementations.

---

_Created by [Muneeb Ansari](https://github.com/Muneeb-Ansari-28)_
