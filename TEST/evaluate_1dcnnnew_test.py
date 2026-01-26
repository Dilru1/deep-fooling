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
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Custom env / policy
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne 
from boatsimulator.core.gl.contextmanager import ContextManager
from cnn_extractor import HistoryCNNExtractor 

# --- Configuration ---
PARENT_DIR = Path("../model/Par_200000_1DCNN")
EVAL_ENV_FILE = "test.json"
OUTPUT_FILE = "Global_Leaderboard_1DCNN.csv"
N_STACK = 4

def get_all_checkpoints(parent_dir):
    candidates = []
    model_pattern = re.compile(r"ppo_sailboat_(\d+)_steps\.zip")
    seed_dirs = sorted(list(parent_dir.glob("seed_*")))
    
    for seed_path in seed_dirs:
        seed_name = seed_path.name
        ckpt_dir = seed_path / "cheakpoints"
        if not ckpt_dir.exists(): continue

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
    
    # 1. SETUP ENV
    # We create the environment params
    for heading, wind in conditions:
        current_params = base_params.copy()
        current_params["target"]["target_headings"] = [heading]
        current_params["wind"]["wind_speeds"] = [wind]
        
        env = None
        model = None
        
        try:
            env = SailboatEnv_consigne(f"Eval", current_params, cm=cm)
            env = FlattenObservation(env)
            env = DummyVecEnv([lambda: env])
            env = VecFrameStack(env, n_stack=N_STACK)
            
            # Load Stats & Model
            env = VecNormalize.load(str(candidate['stats_path']), env)
            env.training = False 
            env.norm_reward = False
            
            # Force CPU to avoid VRAM leaks
            model = PPO.load(str(candidate['model_path']), env=env, device='cpu')
            
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
            # === CRITICAL CLEANUP SECTION ===
            if model is not None:
                del model
            if env is not None:
                env.close()
                del env
            
            # Clear PyTorch Cache (even for CPU it helps clear internal structs)
            torch.cuda.empty_cache() 
            
            # Force Python Garbage Collection
            gc.collect()

    if episodes == 0: return -np.inf
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
    
    # Headless is safer for long loops
    cm = ContextManager(headless=True)
    
    candidates = get_all_checkpoints(PARENT_DIR)
    print(f"Found {len(candidates)} total checkpoints.")

    # Load existing results if we crashed previously (Resume capability)
    results = []
    processed_keys = set()
    
    # Optional: Logic to resume from existing CSV if crash happened
    if Path(OUTPUT_FILE).exists():
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            results = existing_df.to_dict('records')
            for r in results:
                # Create a unique key for Resume Logic
                processed_keys.add(f"{r['seed']}_{r['step']}")
            print(f"Resuming... Found {len(results)} already processed.")
        except:
            print("Could not read existing CSV, starting fresh.")

    for cand in tqdm(candidates, desc="Evaluating Models"):
        # Resume Check
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

    # Final Sort
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="mean_reward", ascending=False)
    df_sorted.to_csv(OUTPUT_FILE, index=False)
    
    print("\nLEADERBOARD TOP 3")
    print(df_sorted.head(3)[["seed", "step", "mean_reward"]])

if __name__ == "__main__":
    main()