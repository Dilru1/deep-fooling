import json
import itertools
import copy
from pathlib import Path
import numpy as np
import pandas as pd

# Environment Imports
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne 
from boatsimulator.core.gl.contextmanager import ContextManager
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize 

# --- Configuration ---
PARENT_DIR = Path("../model/Par_250000_MLP")
EVAL_ENV_FILE = "test.json"
OUTPUT_DIR = Path("Output_CSV")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_single_condition(seed_path, heading, wind, params, cm):
    """
    Runs a single episode for a specific Seed + Heading + Wind combination.
    """
    seed_name = seed_path.name
    
    # 1. Modify params to FORCE this specific condition
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]
    
    # 2. Setup Paths
    model_path = seed_path / "mlp_model_ns128.zip" #changed to best model in this case
    stats_path = seed_path / "vec_normalize_ns128.pkl"
    
    csv_filename = f"test_mlp_{seed_name}_heading_{heading}_wind_{wind}.csv"
    output_path = OUTPUT_DIR / csv_filename

    if output_path.exists():
        print(f"  [Skip] {csv_filename} already exists.")
        return

    # 3. Create Environment with FORCED params
    env = SailboatEnv_consigne(f"Eval {seed_name}", current_params, cm=cm)
    
    # --- FIX START: Get Index Map BEFORE Flattening ---
    # At this point, observation_space is still a Dict
    try:
        keys = list(env.observation_space.spaces.keys())
        index_map = {key: i for i, key in enumerate(keys)}
    except AttributeError:
        # Fallback if the env is somehow already flat or different
        print(f"  [Error] Could not retrieve keys from observation space: {env.observation_space}")
        env.close()
        return
    # --- FIX END ---

    # 4. Apply Wrappers
    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    
    try:
        # Load Stats
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False
        
        # Load Model
        # device='cpu' is added to suppress the GPU warning you saw
        model = PPO.load(str(model_path), env=env, device='cpu') 

    except Exception as e:
        print(f"  [Error] Could not load model/stats for {seed_name}: {e}")
        env.close()
        return

    # 5. Run Episode
    obs = env.reset()
    step_ct = 0
    data_records = []
    
    try:
        while True:
            step_ct += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            # Unnormalize Observation
            original_obs = env.unnormalize_obs(obs)
            flat_obs = original_obs[0]

            # Extract Data using the map we created earlier
            current_cmg = flat_obs[index_map['cmg']]
            current_gs = flat_obs[index_map['ground_speed']]
            current_course = flat_obs[index_map['course_relative']]
            current_heading_rel = flat_obs[index_map['heading_relative']]

            # Extract Position
            current_pos = infos[0].get('current_pos', [0, 0]) 
            pos_dist = infos[0].get('proj_dist_from_start', np.array([np.nan]))
            
            # Recalculation for verification
            calc_cmg = current_gs * np.cos(np.deg2rad(current_course))

            data_records.append({
                "step": step_ct,
                "proj_dist": pos_dist,
                "pos_x": current_pos[0],
                "pos_y": current_pos[1],
                "cmg_env": current_cmg,
                "cmg_calc": calc_cmg,
                "ground_speed": current_gs,
                "course_relative": current_course,
                "heading_relative": current_heading_rel,
                "reward": reward[0],
                "action": action[0]
            })

            if done[0]:
                break
                
        # 6. Save CSV
        df = pd.DataFrame(data_records)
        df.to_csv(output_path, index=False)
        print(f"  [Done] Saved {csv_filename} ({step_ct} steps)")

    except Exception as e:
        print(f"  [Error] Runtime error in {seed_name}: {e}")
    finally:
        env.close()


def main():
    # 1. Load Base Configuration
    if not Path(EVAL_ENV_FILE).exists():
        print(f"Error: {EVAL_ENV_FILE} not found.")
        return
        
    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    # 2. Extract Conditions to Test
    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    
    # Generate all pairs: [(91, 12), (91, 13), ..., (93, 14)]
    combinations = list(itertools.product(headings, winds))
    print(f"Found {len(combinations)} conditions to test per seed.")

    # 3. Setup Context Manager
    cm = ContextManager(headless=True)

    # 4. Find Seeds
    seed_dirs = sorted(list(PARENT_DIR.glob("seed_*")))
    if not seed_dirs:
        print("No seed directories found.")
        return

    # 5. Master Loop
    total_tasks = len(seed_dirs) * len(combinations)
    count = 0
    
    for seed_path in seed_dirs:
        print(f"\n--- Processing {seed_path.name} ---")
        for heading, wind in combinations:
            count += 1
            print(f"({count}/{total_tasks}) Testing Heading: {heading}, Wind: {wind} ...")
            
            run_single_condition(
                seed_path=seed_path,
                heading=heading,
                wind=wind,
                params=base_params,
                cm=cm
            )

    print("\nAll evaluations complete.")

if __name__ == "__main__":
    main()