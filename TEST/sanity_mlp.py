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

    print("Dilruwan")
    print(index_map)
    
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
    last_action = 0.0

    try:
        while True:
            step_ct += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)
            total_reward += reward[0]

            # Unnormalize Observation
            original_obs = env.unnormalize_obs(obs)
            flat_obs = original_obs[0]
            

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
    
    cm = ContextManager(headless=False)
    
    for _, row in checkpoints_df.iterrows():
        for heading, wind in combinations:
            run_single_condition(row, heading, wind, base_params, cm)

    print("\nAll Detailed MLP evaluations complete.")

if __name__ == "__main__":
    main()