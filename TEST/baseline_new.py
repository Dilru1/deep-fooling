import json
import itertools
import copy
from pathlib import Path
import numpy as np
import pandas as pd

# Environment Imports
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne
from boatsimulator.core.gl.contextmanager import ContextManager

# --- Configuration ---
EVAL_ENV_FILE = "test.json"
OUTPUT_DIR = Path("Output_Baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The action that activates the baseline behavior (e.g., 1 = "No Deviation" / PID)
NEUTRAL_ACTION = 1 

def run_baseline_condition(heading, wind, params, cm):
    """
    Runs a single episode using the PID (Neutral Action) for a specific condition.
    """
    # 1. Force the specific condition
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]
    
    csv_filename = f"baseline_heading_{heading}_wind_{wind}.csv"
    output_path = OUTPUT_DIR / csv_filename

    if output_path.exists():
        print(f"  [Skip] {csv_filename} exists.")
        return

    # 2. Initialize Environment (No Wrappers needed for Baseline)
    # We use the raw environment since we don't need to flatten obs for a model
    env = SailboatEnv_consigne("Baseline Env", current_params, cm=cm)
    
    obs, info = env.reset() # Note: gymnasium API returns (obs, info)
    
    step_ct = 0
    data_records = []
    
    try:
        while True:
            step_ct += 1
            
            # --- Execute PID / Neutral Action ---
            step_result = env.step(NEUTRAL_ACTION)
            
            # Handle Gym vs Gymnasium API (4 vs 5 return values)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, infos = step_result
                done = terminated or truncated
            else:
                obs, reward, done, infos = step_result
                # If infos is a dict (not list) in raw env, wrap it to access safely if needed
                if not isinstance(infos, dict): infos = {}

            # --- Extract Data ---
            # In raw env, obs is likely a dictionary already
            cmg = obs.get('cmg', np.nan)
            crs = obs.get('course_relative', np.nan)
            hrs = obs.get('heading_relative', np.nan)
            spd = obs.get('ground_speed', np.nan)
            
            # Info extraction depends on structure (simpler in raw env)
            # Safe access to position from 'infos' dictionary
            current_pos = infos.get('current_pos', [0, 0])
            pos_x = current_pos[0]
            pos_y = current_pos[1]
            pos_dist = infos.get('proj_dist_from_start', np.array([np.nan]))
            
            cmg_calc = spd * np.cos(np.deg2rad(crs))

            data_records.append({
                "step": step_ct,
                "proj_dist": pos_dist,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "cmg_env": cmg,
                "cmg_calc": cmg_calc,
                "ground_speed": spd,
                "course_relative": crs,
                "heading_relative": hrs,
                "reward": reward,
                "action": NEUTRAL_ACTION
            })

            if done:
                break
        
        # 3. Save Results
        df = pd.DataFrame(data_records)
        df.to_csv(output_path, index=False)
        print(f"  [Done] Saved {csv_filename} ({step_ct} steps)")

    except Exception as e:
        print(f"  [Error] Failed heading {heading}, wind {wind}: {e}")
    finally:
        env.close()

def main():
    # 1. Load Config
    if not Path(EVAL_ENV_FILE).exists():
        print("test.json not found.")
        return
    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    # 2. Generate Combinations
    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    combinations = list(itertools.product(headings, winds))
    
    print(f"Running Baseline for {len(combinations)} conditions...")
    
    cm = ContextManager(headless=True)

    for i, (h, w) in enumerate(combinations):
        print(f"Processing {i+1}/{len(combinations)}: Heading {h}, Wind {w}")
        run_baseline_condition(h, w, base_params, cm)

    print("\nBaseline generation complete.")

if __name__ == "__main__":
    main()