import json
import itertools
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import os

# Environment Imports
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne 
from boatsimulator.core.gl.contextmanager import ContextManager
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# --- Configuration ---
EVAL_ENV_FILE = "test.json"
# We save to a 'Baseline' folder to keep it separate from 'MLP' results
BASE_OUTPUT_DIR = Path("Baseline")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The action that activates the baseline PID behavior (neutral steering)
NEUTRAL_ACTION = 1.0 

def run_single_condition(run_id, heading, wind, params, cm):
    """
    Runs a single simulation episode using a constant neutral action (PID Baseline).
    """
    base_name = f"eval_{run_id}_H{heading}_W{wind}"
    csv_path = BASE_OUTPUT_DIR / f"{base_name}.csv"

    if csv_path.exists():
        print(f"  [Skip] {base_name} exists.")
        return

    # 1. Modify current params for this specific run
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]

    # Initialize environment
    env = SailboatEnv_consigne(f"Eval_PID_{run_id}", current_params, cm=cm)
    
    try:
        keys = list(env.observation_space.spaces.keys())
        index_map = {key: i for i, key in enumerate(keys)}
    except AttributeError:
        env.close()
        return

    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])

    # 2. Run Episode with Neutral Action
    obs = env.reset()
    step_ct = 0
    total_reward = 0
    data_records = []
    last_action = NEUTRAL_ACTION

    try:
        while True:
            step_ct += 1
            
            # Use constant neutral action instead of model prediction
            action = np.array([[NEUTRAL_ACTION]]) 
            
            obs, reward, done, infos = env.step(action)
            total_reward += reward[0]

            # In baseline mode (no VecNormalize), obs is already the raw flattened observation
            flat_obs = obs[0]
            info = infos[0]
            
            current_action = action[0][0]
            slew_rate = abs(current_action - last_action)
            last_action = current_action

            # Log exactly the same columns as the MLP script
            data_records.append({
                "step": step_ct,
                "pos_x": info.get('current_pos')[0],
                "pos_y": info.get('current_pos')[1],
                "target_heading": heading,
                "wind_speed": wind,
                "sog_knots": flat_obs[index_map['ground_speed']],
                "cmg_env": flat_obs[index_map['cmg']],
                "xte_error": info.get('ortho_dist_otr', 0),
                "heading_err": info.get('heading_deviation', 0),
                "progress": info.get('proj_dist_from_start', 0),
                "course_relative": flat_obs[index_map['course_relative']],
                "heading_relative": flat_obs[index_map['heading_relative']], 
                "action_offset": current_action,
                "slew_rate": slew_rate,
                "reward": reward[0]
            })

            if done[0]:
                break
                
        # 3. Save Results
        df = pd.DataFrame(data_records)
        df.to_csv(csv_path, index=False)
        print(f"  [Done] Saved {base_name}")

    except Exception as e:
        print(f"  [Error] Runtime error in run {run_id}: {e}")
    finally:
        env.close()

def main():
    if not Path(EVAL_ENV_FILE).exists():
        print(f"Error: {EVAL_ENV_FILE} not found.")
        return
        
    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    combinations = list(itertools.product(headings, winds))
    
    print(f"Starting Baseline Evaluation for {len(combinations)} conditions...")
    
    cm = ContextManager(headless=True)
    
    # We use a simple loop over combinations. 
    # run_id 'PID' helps distinguish these files from MLP runs.
    for heading, wind in combinations:
        run_single_condition("PID", heading, wind, base_params, cm)

    print("\nAll Detailed Baseline evaluations complete.")

if __name__ == "__main__":
    main()