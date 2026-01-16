# RL Sailboat Controller

This project implements a Deep Reinforcement Learning (DRL) agent to control a sailboat using **Stable Baselines 3** and **Gymnasium**. It features custom parallel training environments, a 1D-CNN feature extractor for temporal observations, and configurable reward functions.

## Features

* **Algorithms:** Uses Proximal Policy Optimization (PPO).
* **Policies:**
    * **MLP:** Standard Multi-Layer Perceptron policy.
    * **1D-CNN:** Custom `HistoryCNNExtractor` for processing stacked temporal observations.
* **Parallel Training:** Implements `NonDaemonicSubprocVecEnv` to allow robust multiprocessing with complex simulators.
* **Configuration:** Training parameters (wind, target, rewards) are fully configurable via JSON.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/rl-sailboat-project.git](https://github.com/yourusername/rl-sailboat-project.git)
    cd rl-sailboat-project
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Custom Simulators:**
    > **Note:** This project requires the `boatsgym` and `boatsimulator` libraries. Ensure these are installed in your Python environment or available in your `PYTHONPATH`.

## Configuration

Training parameters are defined in `config/train.json`. Key settings include:
* `global`: Reward function selection (`cmg_reward`), render modes, and step limits.
* `target`: Heading targets and acceptance thresholds.
* `wind`: Wind speed ranges.
* `boat`: Action space parameters (rudder angle).

## Usage

### Training

To train the **MLP** agent (Standard Feed-Forward):
```bash
python src/train_mlp.py