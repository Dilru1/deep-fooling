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
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

# Import custom extractor if needed for loading, though PPO.load usually handles it 
# provided the code is in path.
from cnn_extractor import HistoryCNNExtractor 

# --- Configuration ---
# Update this to point to your 1DCNN model parent directory
PARENT_DIR = Path("../model/Par_250000_1DCNN") 
EVAL_ENV_FILE = "test.json"
OUTPUT_DIR = Path("Output_CSV_1DCNN")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frame Stack Settings
N_STACK = 4

def run_single_condition(seed_path, heading, wind, params, cm):
    """
    Runs a single episode for a specific Seed + Heading + Wind combination
    using the 1D CNN FrameStack configuration.
    """
    seed_name = seed_path.name
    
    # 1. Modify params to FORCE this specific condition
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]
    
    # 2. Setup Paths
    # Adjust filename if your saved model is named differently (e.g. best_model.zip)
    model_path = seed_path / "final_model_ns128.zip" #changed to best model in this case
    stats_path = seed_path / "vec_normalize_ns128.pkl"
    
    csv_filename = f"test_cnn_{seed_name}_heading_{heading}_wind_{wind}.csv"
    output_path = OUTPUT_DIR / csv_filename

    if output_path.exists():
        print(f"  [Skip] {csv_filename} already exists.")
        return

    # 3. Create Environment
    # We pass 'cm' to reuse the context manager (avoids opening/closing windows repeatedly)
    env = SailboatEnv_consigne(f"Eval {seed_name}", current_params, cm=cm)
    
    # --- Get Index Map BEFORE Flattening/Stacking ---
    try:
        keys = list(env.observation_space.spaces.keys())
        index_map = {key: i for i, key in enumerate(keys)}
        n_features = len(keys) # Usually 19
    except AttributeError:
        print(f"  [Error] Could not retrieve keys from observation space.")
        env.close()
        return

    # 4. Apply Wrappers (Order is Critical)
    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    
    # !!! Apply Frame Stacking !!!
    env = VecFrameStack(env, n_stack=N_STACK)
    
    try:
        # Load Normalization Stats
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False
        
        # Load Model
        # device='cpu' forces CPU usage; remove if you want GPU
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

            # --- Data Extraction Logic for FrameStack ---
            
            # 1. Unnormalize
            original_obs = env.unnormalize_obs(obs)
            flat_original_obs = original_obs[0]
            
            # 2. Calculate Offset
            # The observation is stacked: [Frame 1 | Frame 2 | Frame 3 | Frame 4]
            # We want the MOST RECENT frame (Frame 4), which is at the end.
            offset = (N_STACK - 1) * n_features

            # 3. Extract Current Values using Offset
            current_cmg = flat_original_obs[offset + index_map['cmg']]
            current_gs = flat_original_obs[offset + index_map['ground_speed']]
            current_course = flat_original_obs[offset + index_map['course_relative']]
            current_heading_rel = flat_original_obs[offset + index_map['heading_relative']]

            # 4. Extract Position (from infos)
            current_pos = infos[0].get('current_pos', [0, 0]) 
            pos_dist = infos[0].get('proj_dist_from_start', np.array([np.nan]))
            
            # 5. Verification Calculation
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
    
    # Generate all pairs
    combinations = list(itertools.product(headings, winds))
    print(f"Found {len(combinations)} conditions to test per seed.")

    # 3. Setup Context Manager (headless=True for faster batch processing)
    # Set headless=False if you want to watch the rendering (slower)
    cm = ContextManager(headless=True)

    # 4. Find Seeds
    seed_dirs = sorted(list(PARENT_DIR.glob("seed_*")))
    if not seed_dirs:
        print(f"No seed directories found in {PARENT_DIR}.")
        return

    # 5. Master Loop
    total_tasks = len(seed_dirs) * len(combinations)
    count = 0
    
    print(f"Starting evaluation on {len(seed_dirs)} seeds...")
    
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