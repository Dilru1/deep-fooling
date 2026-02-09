# deep-fooling — Reinforcement Learning of a Steering Model for a Foiling Sailboat

<div align="center">
  <img src="assets/ensai-logo.png" alt="ENSAI Logo" height="80" style="margin-right: 20px;"/>
  <img src="assets/nautia-logo.png" alt="Nautia Logo" height="80"/>
</div>

<br/>

## Overview

**deep-fooling** is a Deep Reinforcement Learning (DRL) project developed during the **SMART-DATA (2025–2026)** specialization at **ENSAI**, in collaboration with the startup **Nautia**.

Racing boats in elite solo offshore events like the [Vendée Globe](https://en.wikipedia.org/wiki/Vend%C3%A9e_Globe) rely on high-performance autopilots. Traditional closed-loop control systems, however, often fail to optimise trajectories amidst stochastic environmental changes — forcing sailors to steer manually to remain competitive. With the advent of **hydrofoils**, which reduce drag by lifting the hull, steering complexity has increased further, requiring controllers that can proactively manage both stability and speed.

This project presents a DRL framework that learns optimal steering policies inside a custom **Gymnasium-based** sailing simulator. The agent’s task is formulated as a **Markov Decision Process (MDP)** focused on upwind navigation under variable wind headings (90° – 140°) and wind speeds (10 – 18 knots).

## Key Results

We evaluate and compare **three control strategies**:

| Strategy | Description |
|---|---|
| **PID Baseline** | Traditional proportional–integral–derivative controller |
| **MLP** | Native Multi-Layer Perceptron policy (Stable-Baselines3 default) |
| **1D CNN** | Novel 1D Convolutional Neural Network feature extractor |

**Highlights:**

- The **1D CNN** achieved a peak *Course Made Good* (CMG) of **24.59 knots** under stress-test conditions (95° heading, 20 kts wind).
- It outperformed the PID baseline in the majority of in-distribution scenarios, with improvements of up to **+5.27 %**.
- The MLP struggled with stability in high-wind regimes, often under-performing the PID by significant margins.
- The 1D CNN showed some vulnerability to *Out-of-Bounds* failures at light-wind boundaries (10 kts), but proved to be the most robust architecture for high-speed foiling overall.

> 📄 **Paper:** [Read the full report](Report/main.pdf)

## Project Structure

```
deep-fooling/
├── cnn_extractor.py               # 1D CNN feature extractor (HistoryCNNExtractor)
├── minimal_train_1dcnn_par_seed.py # Training script — 1D CNN policy (parallel, seeded)
├── minimal_train_par_seed.py       # Training script — MLP policy (parallel, seeded)
├── parallel_env.py                 # Non-daemonic SubprocVecEnv for parallel envs
├── rewards.py                      # CMG-based reward function
├── event.py                        # Event definitions
├── environment.json                # Evaluation environment configuration
├── train.json                      # Training environment configuration
├── notebooks/
│   ├── plots.ipynb                 # Visualisation & analysis notebook
│   ├── 1DCNN/                      # 1D CNN evaluation CSVs
│   ├── MLP/                        # MLP evaluation CSVs
│   └── Baseline/                   # PID baseline evaluation CSVs
├── TEST/                           # Testing & evaluation scripts
├── Report/                         # LaTeX source for the paper
└── assets/                         # Logos and images
```

## Architecture

### Reward Function

The agent maximises **Course Made Good (CMG)** — the projected boat speed along the target heading — with an out-of-bounds penalty when the relative heading exceeds ±45°:

$$r_t = \text{CMG}_t - \mathbb{1}\left[|\Delta\theta| \geq 45°\right] \cdot c_{\text{oob}}$$

### 1D CNN Feature Extractor

The custom `HistoryCNNExtractor` reshapes stacked observations into a temporal sequence and applies two 1D convolution layers, allowing the policy to extract short-term temporal patterns from wind fluctuations:

```
Input (flat) → Reshape (stack × features) → Conv1D(32) → ReLU → Conv1D(64) → ReLU → Flatten → Linear(64)
```

### Training

- **Algorithm:** PPO (Proximal Policy Optimization) via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- **Observation stacking:** `VecFrameStack` (3 frames)
- **Normalisation:** `VecNormalize`
- **Parallelism:** Custom `NonDaemonicSubprocVecEnv` for multi-process rollout collection
- **Learning rate:** Linear schedule with configurable initial and final values
- **Timesteps:** 250,000

## Getting Started

### Prerequisites

- Python 3.10+
- Access to the Nautia `boatsgym` / `boatsimulator` packages

### Installation

```bash
git clone https://github.com/Dilru1/deep-fooling.git
cd deep-fooling
pip install -r requirments.txt
```

### Training

```bash
# Train with the 1D CNN policy
python minimal_train_1dcnn_par_seed.py

# Train with the default MLP policy
python minimal_train_par_seed.py
```

### Evaluation

Evaluation scripts are located in the `TEST/` directory. See `TEST/evaluate_1dcnn.py` and `TEST/evaluate_mlp.py`.

## Acknowledgements

This project was developed at **ENSAI** as part of the SMART-DATA specialization (2025–2026), in partnership with **Nautia**.
