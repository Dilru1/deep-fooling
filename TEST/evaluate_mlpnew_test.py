import os
# --- CRITICAL FIX: Force Software Rendering ---
# This prevents Segmentation Faults on headless servers
#os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
##os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
#os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

import json
import itertools
import re
import pandas as pd
import numpy as np
import gc  # <--- IMPORT GARBAGE COLLECTOR
import torch # <--- IMPORT TORCH FOR CLEANUP
from pathlib import Path
from tqdm.auto import tqdm

# Gymnasium & SB3
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Custom env / policy
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne 
from boatsimulator.core.gl.contextmanager import ContextManager
# Ensure this import works in your file structure
from cnn_extractor import HistoryCNNExtractor 

# --- Configuration ---
PARENT_DIR = Path("../model/Par_200000_MLP") 
EVAL_ENV_FILE = "test.json"
OUTPUT_FILE = "Global_Leaderboard_MLP.csv"
# N_STACK is not used for MLP usually, but keeping variable to avoid errors if referenced
N_STACK = 4 

def get_all_checkpoints(parent_dir):
    candidates = []
    model_pattern = re.compile(r"ppo_sailboat_(\d+)_steps\.zip")
    seed_dirs = sorted(list(parent_dir.glob("seed_*")))
    
    for seed_path in seed_dirs:
        seed_name = seed_path.name
        ckpt_dir = seed_path / "cheakpoints"
        if not ckpt_dir.exists():
            continue

        for model_path in ckpt_dir.glob("*.zip"):
            match = model_pattern.search(model_path.name)
            if match:
                step = int(match.group(1))
                stats_name = f"ppo_sailboat_vecnormalize_{step}_steps.pkl"
                stats_path = ckpt_dir / stats_name
                
                if stats_path.exists():
                    candidates.append({
                        "seed": seed_name,
                        "step": step,
                        "model_path": model_path,
                        "stats_path": stats_path
                    })
    
    candidates.sort(key=lambda x: (x['seed'], x['step']))
    return candidates

def evaluate_candidate(candidate, conditions, base_params, cm):
    total_reward = 0.0
    episodes = 0
    
    # Run through all conditions (Wind x Heading)
    for heading, wind in conditions:
        current_params = base_params.copy()
        current_params["target"]["target_headings"] = [heading]
        current_params["wind"]["wind_speeds"] = [wind]
        
        env = None
        model = None
        
        try:
            # 1. Create Environment
            env = SailboatEnv_consigne(f"Eval", current_params, cm=cm)
            env = FlattenObservation(env)
            env = DummyVecEnv([lambda: env])
            
            # Note: No VecFrameStack for MLP usually
            
            # 2. Load Stats
            env = VecNormalize.load(str(candidate['stats_path']), env)
            env.training = False     
            env.norm_reward = False 
            
            # 3. Load Model (Force CPU to prevent VRAM accumulation)
            model = PPO.load(str(candidate['model_path']), env=env, device='cpu')
            
            # 4. Run Episode
            obs = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
            
            total_reward += episode_reward
            episodes += 1
            
        except Exception as e:
            print(f"\n[Error] Failed on {candidate['seed']} step {candidate['step']}: {e}")
            return -np.inf 
        
        finally:
            # === MEMORY CLEANUP ===
            if model is not None:
                del model
            if env is not None:
                env.close()
                del env
            
            # Clear PyTorch internal cache
            torch.cuda.empty_cache()
            # Force Python Garbage Collection
            gc.collect()

    if episodes == 0:
        return -np.inf

    return total_reward / episodes

def main():
    if not Path(EVAL_ENV_FILE).exists():
        print(f"Error: {EVAL_ENV_FILE} not found.")
        return

    with open(EVAL_ENV_FILE) as f:
        base_params = json.load(f)

    headings = base_params["target"]["target_headings"]
    winds = base_params["wind"]["wind_speeds"]
    conditions = list(itertools.product(headings, winds))
    
    cm = ContextManager(headless=True)
    
    candidates = get_all_checkpoints(PARENT_DIR)
    print(f"Found {len(candidates)} total checkpoints.")

    # --- RESUME LOGIC ---
    results = []
    processed_keys = set()
    
    if Path(OUTPUT_FILE).exists():
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            results = existing_df.to_dict('records')
            for r in results:
                # Create unique ID for what we have already done
                processed_keys.add(f"{r['seed']}_{r['step']}")
            print(f"Resuming... Found {len(results)} already processed.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting fresh.")

    # --- MAIN LOOP ---
    for cand in tqdm(candidates, desc="Evaluating Models"):
        # Skip if already done
        unique_key = f"{cand['seed']}_{cand['step']}"
        if unique_key in processed_keys:
            continue

        avg_reward = evaluate_candidate(cand, conditions, base_params, cm)
        
        results.append({
            "seed": cand['seed'],
            "step": cand['step'],
            "mean_reward": avg_reward,
            "model_path": str(cand['model_path'])
        })
        
        # Intermediate Save
        pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

    # --- FINALIZE ---
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="mean_reward", ascending=False)
    df_sorted.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "="*40)
    print("           LEADERBOARD TOP 3")
    print("="*40)
    if not df_sorted.empty:
        print(df_sorted.head(3)[["seed", "step", "mean_reward"]])
        best_model = df_sorted.iloc[0]
        print(f"\nWINNER: {best_model['seed']} at step {best_model['step']}")
        print(f"Reward: {best_model['mean_reward']:.2f}")

if __name__ == "__main__":
    main()