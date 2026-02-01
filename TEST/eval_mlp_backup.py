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
from stable_baselines3.common.vec_env import VecNormalize 

# --- Configuration ---
CHECKPOINT_CSV = "best_checkpoints_found.csv"
EVAL_ENV_FILE = "test.json"
BASE_OUTPUT_DIR = Path("MLP")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FLAG = "Par_250000_MLP" 

def run_single_condition(checkpoint_row, heading, wind, params, cm):
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

    # 2. Extract Static Simulation Metadata
    # These remain constant for the entire episode
    sim_p = params.get('simulation_params', {})
    wave_amplitudes = sim_p.get('external_wave_amplitudes', [0])
    max_wave_amp = max(wave_amplitudes) if isinstance(wave_amplitudes, list) else wave_amplitudes
    foil_rake = sim_p.get('kdf_rakes', [0, 0, 0])[2] # Usually the 3rd index

    # 3. Modify current params
    current_params = copy.deepcopy(params)
    current_params["target"]["target_headings"] = [heading]
    current_params["wind"]["wind_speeds"] = [wind]

    env = SailboatEnv_consigne(f"Eval {run_id}", current_params, cm=cm)
    
    try:
        keys = list(env.observation_space.spaces.keys())
        index_map = {key: i for i, key in enumerate(keys)}
    except AttributeError:
        env.close()
        return

    env = FlattenObservation(env)
    env = DummyVecEnv([lambda: env])
    
    try:
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(model_path), env=env, device='cpu') 
    except Exception as e:
        print(f"  [Error] Load failed: {e}")
        env.close()
        return

    # 4. Run Episode
    obs = env.reset()
    step_ct = 0
    total_reward = 0
    data_records = []
    
    try:
        while True:
            step_ct += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += reward[0]

            # Unnormalize Observation
            original_obs = env.unnormalize_obs(obs)
            flat_obs = original_obs[0]
            
            # INFO Dictionary Extraction (Sensory Inputs)
            info = infos[0]
            current_pos = info.get('current_pos', [0, 0])

            data_records.append({
                "step": step_ct,
                # --- Static Metadata ---
                "target_heading": heading,
                "wind_speed": wind,
                "wave_amp_max": max_wave_amp,
                "foil_rake_init": foil_rake,
                # --- Dynamic Navigation ---
                "pos_x": current_pos[0],
                "pos_y": current_pos[1],
                "sog_knots": info.get('sog', 0), # Speed Over Ground
                "cmg_deg": flat_obs[index_map['cmg']], # Course Made Good
                "gs_ms": flat_obs[index_map['ground_speed']], # Ground Speed m/s
                # --- Control & Effort ---
                "action_os": action[0], # The RL agent's offset
                "rudder_angle": info.get('rudder_angle', 0),
                "rudder_torque": info.get('rudder_torque', 0),
                "keel_angle": info.get('keel_angle', 0),
                # --- Environmental ---
                "twa": info.get('twa', 0), # True Wind Angle
                "awa": info.get('awa', 0), # Apparent Wind Angle
                "reward": reward[0],
            })

            if done[0]:
                break
                
        # 5. Save Results
        df = pd.DataFrame(data_records)
        df.to_csv(csv_path, index=False)
        
        with open(txt_path, "w") as f:
            f.write(f"Detailed Evaluation: {run_id}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Target Heading: {heading} | Wind: {wind} | Max Wave: {max_wave_amp}\n")
            f.write(f"Final SOG (Avg): {df['sog_knots'].mean():.2f} knots\n")
            f.write(f"Total Steps: {step_ct} | Total Reward: {total_reward:.2f}\n")
            f.write(f"Avg Rudder Torque: {df['rudder_torque'].mean():.2f} N\n")
            f.write(f"Checkpoint: {model_path.name}\n")

        print(f"  [Done] Saved {base_name}")

    except Exception as e:
        print(f"  [Error] Runtime error: {e}")
    finally:
        env.close()

def main():
    if not Path(CHECKPOINT_CSV).exists():
        print(f"Error: {CHECKPOINT_CSV} not found.")
        return
        
    df = pd.read_csv(CHECKPOINT_CSV)
    checkpoints_df = df[df['Architecture'] == TARGET_FLAG]
    
    if checkpoints_df.empty:
        print(f"No models found for {TARGET_FLAG}")
        return

    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    combinations = list(itertools.product(headings, winds))
    
    cm = ContextManager(headless=True)
    
    for _, row in checkpoints_df.iterrows():
        for heading, wind in combinations:
            run_single_condition(row, heading, wind, base_params, cm)

    print("\nAll Detailed MLP evaluations complete.")

if __name__ == "__main__":
    main()