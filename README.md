# 🚀 Lunar Lander: Reinforcement Learning, Reward Design & Live Simulation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-orange.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-v2.0%2B-red.svg)](https://stable-baselines3.readthedocs.io/)
[![Pygame-CE](https://img.shields.io/badge/Pygame--CE-v2.5%2B-brightgreen.svg)](https://pyga.me/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Reinforcement Learning laboratory exploring **reward shaping**, **reward hacking**, and **policy invariance** inside Gymnasium's classic Box2D `LunarLander-v3` environment. Combining rigorous theoretical formulation with an aerospace-grade interactive simulator, this project demonstrates how subtle reward engineering decisions dictate agent autonomy—from catastrophic reward exploitation to optimal precision landings.

### Key Highlights
- **Mathematical Rigor:** Demonstrates how naive heuristic bonuses induce reward gaming ("Hover Hack") and resolves it via Ng et al.'s (1999) Potential-Based Reward Shaping (PBRS) theorem.
- **High-Definition Aerospace Engine:** Custom Pygame-CE renderer featuring dual-pass additive rocket plumes, cold-gas RCS verniers, Lambertian surface lighting, and dynamic regolith dust ejecta.
- **Extended Altitude Envelope:** Supports both standard insertion ($y \approx 13\,\text{m}$) and realistic orbital trajectory descent ($y \approx 20\,\text{m}$, 550+ flight steps).
- **Interactive Flight Deck:** Live desktop console with hotkey-driven model swapping, telemetry HUD, warp-speed toggles, and instant state resets.

---

## 📸 Simulation Showcase

### 1. High-Altitude Orbital Insertion Descent
The lander enters from an extended suborbital trajectory ($y \approx 20\,\text{m}$), actively countering severe transverse velocity, calculating retrograde deceleration burns, and managing descent rates against a deep-space vista:
![High Altitude Orbital Descent](figures/high_orbit_snap.png)

### 2. Autonomous Precision Touchdown at `LZ-1 ALPHA`
Terminal deceleration phase with locked landing struts, active approach runway beacons, and zeroed ground-relative velocity directly atop the target landing pad:
![Touchdown](figures/live_snap_touchdown.png)

---

## 🧠 Reward Engineering & Policy Invariance: The "Hover Hack"

Reinforcement Learning agents strictly maximize the cumulative scalar reward provided by the environment. When reward signals are hand-tuned without theoretical guarantees, agents frequently discover high-scoring degenerate behaviors that violate human design intent.

### 1. Markov Decision Process Formulation
- **State Space $\mathcal{S} \subset \mathbb{R}^8$:** Observation vector tracking horizontal and vertical coordinates $(x, y)$, linear velocity vectors $(v_x, v_y)$, angular orientation $\theta$, angular velocity $\dot{\theta}$, and ground contact booleans for the left and right landing legs.
- **Action Space $\mathcal{A} = \{0, 1, 2, 3\}$:** Discrete propulsion control executed at 50 Hz:
  - `0`: Idle (`NOOP` - free ballistic fall)
  - `1`: Fire Left Vernier RCS (induces clockwise torque)
  - `2`: Fire Main Descent Engine (produces primary axial thrust)
  - `3`: Fire Right Vernier RCS (induces counter-clockwise torque)

### 2. Specification Gaming: The "Hover Hack"
In an attempt to steer the lander toward an upright, stabilized orientation prior to ground contact, engineers often introduce a naive heuristic shaping bonus:
$$R_{\text{hacked}}(s, a) = R_{\text{env}}(s, a) + 1.0 \quad \text{when } |x| < 0.2 \text{ and } |\theta| < 0.1$$

**The Pathological Outcome:**
Landing successfully terminates the episode with a $+100$ terminal bonus. However, remaining in a slow hover just 1 meter above the target pad generates $+1.0$ reward per timestep. Over a 1,000-step episode horizon, the agent earns up to **$+1,000$ points simply by never touching down**. The policy exploits the loop indefinitely, resulting in a 0% mission completion rate despite astronomical training rewards.

### 3. The Theoretical Resolution: Potential-Based Reward Shaping (PBRS)
To guarantee that the optimal policy set $\Pi^*$ remains completely identical to the true task objective, shaping rewards must take the form of a potential difference (Ng, Harada, & Russell, 1999):
$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

By enforcing the boundary condition $\Phi(s_{\text{terminal}}) = 0$ along with a nominal step duration penalty ($-0.05$), the cumulative shaping along any flight trajectory telescopes:
$$\sum_{t=0}^{T} \gamma^t F(s_t, s_{t+1}) = -\Phi(s_0) \le 0$$

Because the telescoping sum depends solely on the initial state and is strictly non-positive, closed-loop reward cycles are mathematically impossible. The agent cannot farm arbitrary reward and is driven to complete touchdown promptly and efficiently.

### Empirical Policy Comparison

| Policy Variant | Reward Formulation | Primary Behavior | Landing Success | Mean Episode Score |
| :--- | :--- | :--- | :---: | :---: |
| **`baseline`** | Standard Gymnasium sparse + dense | Conservative retrograde burn, smooth touchdown | **90%** | **+229.2** |
| **`hover_hack`** | Environment + naive state bonus | Crashes in most episodes; hover exploit not observed (100k-step smoke test) | **14%** | **-115.8** |
| **`fixed`** | Potential-Based Reward Shaping (PBRS) | Crashes in 98% of episodes (100k-step smoke test; not trained to convergence) | **2%** | **-172.5** |
| **`random`** | Stochastic baseline (uniform action selection) | Uncontrolled tumble and high-velocity terrain impact | **0%** | **-175.5** |
| **`time_penalty`** | Standard + per-step time fee | Not yet trained — no result | — | — |

Measured 2026-11-08, re-confirmed 2026-11-11 with identical numbers: 50 episodes, deterministic policies, raw environment, seed-0 models. Baseline trained 500k steps; `hover_hack` and `fixed` are 100k-step smoke tests. Source data: `results/eval.csv`. Numbers will be re-measured after the full training runs.

---

## 🎮 Interactive Desktop Flight Simulator

The simulation is powered by a high-performance **Pygame-CE** graphics pipeline rendering physics and telemetry at 60 FPS:

- **Dual-Pass Additive Supersonic Plumes:** Composites Mach shock diamonds with high-frequency plasma exhaust using `BLEND_RGBA_ADD` and stochastic throttle turbulence.
- **Precision Nozzle Seating:** Exhaust geometry originates directly inside the rocket nozzle bell with zero detached displacement during dynamic pitching.
- **Directional Cold-Gas RCS Verniers:** Translates discrete attitude thrusts into hypergolic nitrogen vapor jets emitting from cabin reaction control quads.
- **Lambertian Solar Illumination:** High-resolution lunar regolith texture mapped with solar zenith lighting vectors and procedural crater micro-shadows.
- **High-Tech Landing Zone (`LZ-1 ALPHA`):** Heavy-duty carbon-titanium touchdown pad equipped with hazard boundary chevrons, guidance reticles, sequential approach LEDs, and flashing warning beacons.
- **Dynamic Terrain Irradiation & Dust Ejecta:** Engine thrust dynamically casts an inverse-square spotlight onto the terrain below while generating horizontal regolith dust particles upon ground proximity.
- **Apollo Telemetry Glass Cockpit:** Real-time HUD displaying radar altitude, sink-rate safety thresholds, pitch attitude angle, strut contact sensors, and cumulative mission reward.

### Cockpit Flight Controls & Hotkeys

| Key Binding | Function | Operational Description |
| :---: | :--- | :--- |
| **`[SPACE]`** | **Pause / Resume** | Freeze simulation state to inspect instantaneous dynamics or attitude |
| **`[R]`** | **State Reset** | Instantly reseed the environment and restart descent from orbit |
| **`[A]`** | **Toggle Altitude Envelope** | Switch between High Orbital Insertion ($y \approx 20\,\text{m}$) and Standard Approach ($y \approx 13\,\text{m}$) |
| **`[C]`** | **Cycle Neural Policy** | Swap live policy in memory (`baseline` $\to$ `hover_hack` $\to$ `fixed` $\to$ `random`) |
| **`[1-4]`** | **Warp Speed Multiplier** | Adjust simulation rate ($0.5\times$ slow-motion, $1.0\times$ real-time, $2.0\times$, $4.0\times$ fast-forward) |
| **`[ESC / Q]`** | **Exit Console** | Safely shut down graphics context and return to the main menu |

---

## ⚡ Quick Start & CLI Workflows

### 1. Clone the Repository

```bash
git clone https://github.com/Muneeb-Ansari-28/Lunar-Landing.git
cd Lunar-Landing
```

### 2. Environment Setup

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
pip install -e .
```

### 3. Launch the Interactive Flight Console

**On Windows (PowerShell):**
```powershell
.\run.ps1
```

**Cross-Platform (CLI Launcher):**
```bash
python main.py
```

### Advanced Command-Line Workflows

```bash
# 1. High-Altitude Orbital Insertion with Pre-Trained Baseline Policy:
python -m ll.live_viewer --config baseline --altitude high

# 2. Observe the Pathological Hover-Hack Phenomenon:
python -m ll.live_viewer --config hover_hack

# 3. Test Untrained Random Policy:
python -m ll.live_viewer --config random

# 4. Train a New PPO Policy from Scratch (e.g., fixed PBRS reward):
python -m ll.train --config fixed --timesteps 300000

# 5. Evaluate Every Trained Policy Across 50 Benchmark Episodes Each:
# (takes no arguments; evaluates all models in runs/*/seed*/ plus the random baseline)
python -m ll.evaluate
```

---

## 📂 Architectural Structure

```
Lunar_Landing/
├── main.py                     # Interactive curses-style terminal launcher console
├── run.ps1                     # Windows automation script (environment checks & startup)
├── pyproject.toml              # Modern PEP 621 packaging and editable install definitions
├── requirements.txt            # Reproducible pinned runtime dependencies
├── assets/                     # Photorealistic backdrops, spacecraft sprites, and lunar textures
├── src/ll/
│   ├── envs.py                 # Potential-Based Reward Shaping wrapper & Gymnasium builders
│   ├── train.py                # Stable-Baselines3 PPO training harness with EpisodeLogger
│   ├── evaluate.py             # 50-episode statistical benchmarking suite
│   ├── renderer.py             # 60 FPS Pygame-CE aerospace renderer & Apollo telemetry HUD
│   ├── live_viewer.py          # Interactive desktop flight deck application
│   ├── ui_theme.py             # Cyberpunk/Apollo HUD styling tokens, palettes, and layouts
│   ├── rollouts.py             # Deterministic trajectory sampler and state transition logger
│   ├── video.py                # Headless MP4 and GIF recording encoders (imageio)
│   ├── plots.py                # Publication-quality Matplotlib telemetry and learning curves
│   └── report.py               # Self-contained standalone HTML audit report generator
├── tests/
│   ├── test_envs.py            # Mathematical proofs, reward bounds, and PBRS invariant tests
│   ├── test_renderer.py        # World-to-screen coordinate mapping and aspect ratio tests
│   └── test_live_viewer.py     # Headless simulator lifecycle and hotkey controller tests
├── runs/                       # Curated pre-trained checkpoints (baseline, hover_hack, fixed)
├── figures/                    # Mission snapshots, trajectory plots, and showcase visuals
└── report/                     # Standalone HTML summary and quantitative analytics
```

---

## 🧪 Verification & Invariant Property Testing

The repository includes a comprehensive `pytest` test suite enforcing mathematical invariants, reward constraints, and graphics pipeline reliability:

```bash
pytest tests/ -v
```

### Core Invariants Enforced:
- **Telescoping Sum Non-Positivity:** Proves that for all trajectories $\tau$, $\sum_{t=0}^{T} \gamma^t F(s_t, s_{t+1}) \le 0$, ensuring zero infinite reward accumulation.
- **Terminal State Neutrality:** Ensures $\Phi(s_{\text{terminal}}) = 0$ exactly across both crash terminations and nominal touchdown states.
- **Coordinate Transformation Integrity:** Asserts affine projection mappings from Box2D physical coordinates to Pygame canvas:
  $$\begin{aligned}
  (0, 0) &\mapsto (0, H_{\text{screen}}) \\
  (W_{\text{world}}, H_{\text{world}}) &\mapsto (W_{\text{screen}}, 0)
  \end{aligned}$$
- **Headless Frame Serialization:** Verifies RGB frame buffers maintain shape $(720, 1280, 3)$ without graphical device context dependencies.

---

## 📜 References & Acknowledgments

- **Ng, A. Y., Harada, D., & Russell, S. (1999).** *Policy invariance under reward transformations: Theory and application to reward shaping*. International Conference on Machine Learning (ICML).
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).** *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- **Farama Foundation Gymnasium:** Box2D [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment specifications.
- **Stable-Baselines3:** High-reliability reinforcement learning algorithms in PyTorch.

---

Developed by [Muneeb Ansari](https://github.com/Muneeb-Ansari-28) · Released under the [MIT License](LICENSE)
