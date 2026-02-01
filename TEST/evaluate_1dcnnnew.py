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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

# --- Configuration ---
CHECKPOINT_CSV = "best_checkpoints_found.csv"
EVAL_ENV_FILE = "test.json"
# Updated directory name to match the architecture flag
BASE_OUTPUT_DIR = Path("1DCNN")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FLAG = "Par_250000_1DCNN" 
N_STACK = 4

def run_single_condition(checkpoint_row, heading, wind, params, cm):
    """
    Runs an episode for the 1DCNN champion using identical headers to the MLP version.
    """
    run_id = checkpoint_row['ID']
    
    # 1. Setup Paths
    model_path = Path(checkpoint_row['Checkpoint_Path'])
    stats_name = model_path.name.replace(".zip", "").replace("ppo_sailboat_", "ppo_sailboat_vecnormalize_") + ".pkl"
    stats_path = model_path.parent / stats_name

    if not stats_path.exists():
        print(f"  [Error] Stats file not found: {stats_path}")
        return
    
    base_name = f"eval_{run_id}_H{heading}_W{wind}"
    csv_path = BASE_OUTPUT_DIR / f"{base_name}.csv"
    txt_path = BASE_OUTPUT_DIR / f"{base_name}.txt"

    if csv_path.exists():
        print(f"  [Skip] {base_name} exists.")
        return

    # 2. Extract Static Simulation Metadata (from params)
    sim_p = params.get('simulation_params', {})
    wave_amplitudes = sim_p.get('external_wave_amplitudes', [0])
    max_wave_amp = max(wave_amplitudes) if isinstance(wave_amplitudes, list) else wave_amplitudes
    foil_rake = sim_p.get('kdf_rakes', [0, 0, 0])[2] 

    # 3. Setup Environment
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]

    env = SailboatEnv_consigne(f"Eval_{run_id}", current_params, cm=cm)
    
    try:
        keys = list(env.observation_space.spaces.keys())
        index_map = {key: i for i, key in enumerate(keys)}
        n_features = len(keys)
    except AttributeError:
        env.close()
        return

    # Apply Wrappers (Critical order for 1DCNN)
    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=N_STACK)
    
    try:
        # Load Stats & Model
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(model_path), env=env, device='cpu') 
        print(f"  [Loaded] Model: {model_path.name}")
    except Exception as e:
        print(f"  [Error] Load failed: {e}")
        env.close()
        return

    # 4. Run Episode
    obs = env.reset()
    step_ct = 0
    total_reward = 0
    data_records = []
    last_action = 0.0
    
    try:
        while True:
            step_ct += 1
            # 'action' is the agent's adjustment (os value)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += reward[0]

            # Unnormalize and handle FrameStack offset
            original_obs = env.unnormalize_obs(obs)
            flat_obs = original_obs[0]
            # offset jumps to the most recent frame in the stack
            offset = (N_STACK - 1) * n_features
            
            info = infos[0]
            current_pos = info.get('current_pos', [0, 0])
            current_action = action[0][0] if isinstance(action[0], (list, np.ndarray)) else action[0]
            slew_rate = abs(current_action - last_action)
            last_action = current_action

            # 5. Record Data with headers identical to MLP case
            data_records.append({
            "step": step_ct,
            "pos_x": current_pos[0],
            "pos_y": current_pos[1],
        
            # --- Static Metadata ---
            "target_heading": heading,
            "wind_speed": wind,

            
            # --- Navigation Performance (Using offset for current state) ---
            "sog_knots": flat_obs[offset + index_map['ground_speed']], 
            "cmg_deg": flat_obs[offset + index_map['cmg']], 
            "xte_error": info.get('ortho_dist_otr', 0),
            "heading_err": info.get('heading_deviation', 0),
            "progress": info.get('proj_dist_from_start', 0),   # Net distance made
            "course_relative": flat_obs[offset + index_map['course_relative']],
            "heading_relative": flat_obs[offset + index_map['heading_relative']],


            # --- Agent Behavior ---
            "action_offset": current_action,
            "slew_rate": slew_rate, #stability indicator
            "reward": reward[0]
            })

            if done[0]: break
                
        # 6. Save Outputs
        df = pd.DataFrame(data_records)
        df.to_csv(csv_path, index=False)
        
        #with open(txt_path, "w") as f:
        #    f.write(f"Detailed Evaluation: {run_id} (1DCNN)\n")
        #    f.write(f"{'='*50}\n")
        #    f.write(f"Architecture: {TARGET_FLAG}\n")
        #    f.write(f"Target Heading: {heading} | Wind: {wind}\n")
        #    f.write(f"Final SOG (Avg): {df['sog_knots'].mean():.2f} knots\n")
        #    f.write(f"Total Steps: {step_ct} | Total Reward: {total_reward:.2f}\n")
        #    f.write(f"Avg Action Offset: {df['action_offset'].mean():.4f}\n")
        #    f.write(f"Checkpoint: {model_path.name}\n")

        #print(f"  [Saved] {base_name} CSV/TXT")

    except Exception as e:
        print(f"  [Error] Runtime error: {e}")
    finally:
        env.close()

def main():
    if not Path(CHECKPOINT_CSV).exists():
        print(f"Error: {CHECKPOINT_CSV} not found.")
        return
    
    df = pd.read_csv(CHECKPOINT_CSV)
    
    # Filter for 1DCNN entries only
    checkpoints_df = df[df['Architecture'] == TARGET_FLAG]

    if checkpoints_df.empty:
        print(f"No models found matching: {TARGET_FLAG}")
        return
    
    print(f"Target flag: {TARGET_FLAG} | Processing {len(checkpoints_df)} seeds.")

    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    combinations = list(itertools.product(headings, winds))

    cm = ContextManager(headless=True)

    for _, row in checkpoints_df.iterrows():
        print(f"\n--- Evaluating Champion: {row['ID']} ---")
        for heading, wind in combinations:
            run_single_condition(row, heading, wind, base_params, cm)

    print(f"\nEvaluation completed for {TARGET_FLAG}.")

if __name__ == "__main__":
    main()