import json
from pathlib import Path

import numpy as np
import pandas as pd
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne 
from boatsimulator.core.gl.contextmanager import ContextManager
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack # Import VecFrameStack

from cnn_extractor import HistoryCNNExtractor

MODEL_DIR = Path("Output")
MODEL_DIR.mkdir(parents=True, exist_ok=True)  
# --- Configuration for Experiment --

#BestMODEL_NAME = "../model/Par_100000_1DCNN/seed_0/best_model/best_model.zip"

MODEL_NAME = "../model/Par_100000_1DCNN/seed_2/final_model_ns128.zip"
EVAL_ENV_FILE = "test.json"

with open(EVAL_ENV_FILE) as f:
    params = json.load(f)

OUTPUT_CSV = f"test_1dcnn_s1.csv"

#params["train"] = "False"
#print(f'Successfully loaded "{EVAL_ENV_FILE}" and set "train": "False".')

# Create the openGL context manager for visual rendering.
cm = ContextManager(headless=False)

# Create the environment
print("Creating Sailboat Environment...")
env = SailboatEnv_consigne("Test env.", params, cm=cm)

keys = list(env.observation_space.spaces.keys())
index_map = {key: i for i, key in enumerate(keys)}
print(index_map)

# --- Load the wrappers ---
env = FlattenObservation(env)
env = DummyVecEnv([lambda:env])

# !!! FIX: Apply Frame Stacking to match the training shape (19 * 5 = 95) !!!
# If you are unsure if it was 5, the math (95/19) confirms it.
env = VecFrameStack(env, n_stack=4)

STATS_PATH = "../model/Par_100000_1DCNN/seed_1/vec_normalize_ns128.pkl"            # new
print(f"Loading normalization stats from {STATS_PATH}...") # new

env = VecNormalize.load(STATS_PATH, env)                   # new
env.training = False                                       # new
env.norm_reward = False                                    # new


# --- Load the trained model ---
model = PPO.load(MODEL_NAME, env=env)
        

data_records = [] # Data storage

# Reset the environment to start Episode 1
obs = env.reset() # VecEnv reset only returns obs
step_ct = 0
#print("\n--- Running Simulation ---")
#print("  (CMG = from env, Calc = from formula, G.Speed = Ground Speed, C.Err = Course Error)")
try:
    while True:  # Loop until episode ends
        step_ct += 1
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = env.step(action)

        original_obs = env.unnormalize_obs(obs)                                     # new
        flat_original_obs = original_obs[0]                                         # new
        
        # Calculate the offset for the most recent frame
        # (n_stack - 1) * n_features = (5 - 1) * 19 = 76
        n_features = 19
        offset = (4 - 1) * n_features

        #current_cmg = flat_original_obs[index_map['cmg']]                           # new
        #current_course_relative = flat_original_obs[index_map['course_relative']]   # new
        #current_ground_speed = flat_original_obs[index_map['ground_speed']]         # new
        #current_heading_relative = flat_original_obs[index_map['heading_relative']] # new

        # ---Framstack seen .. Add offset to get CURRENT values
        current_cmg = flat_original_obs[offset + index_map['cmg']]
        current_course_relative = flat_original_obs[offset + index_map['course_relative']]
        current_ground_speed = flat_original_obs[offset + index_map['ground_speed']]
        current_heading_relative = flat_original_obs[offset + index_map['heading_relative']]

        # new: ready for tracking current position for each step
        current_pos = infos[0].get('current_pos', [0, 0]) 
        pos_x = current_pos[0]
        pos_y = current_pos[1]

        # --- FIX: Read data from the flattened 'obs' array by index ---
        # The obs variable from DummyVecEnv is a list [array], so we get obs[0]
        flat_obs_array = obs[0]

        
        # Get data based on the alphabetical order of keys:
        # 'cmg' is at index 3
        # 'course_relative' is at index 5
        # 'ground_speed' is at index 6
        # 'heading_relative' is at index 7
        
        #current_cmg = flat_obs_array[index_map['cmg']]
        #current_course_relative = flat_obs_array[index_map['course_relative']]
        #current_ground_speed = flat_obs_array[index_map['ground_speed']]
        #current_heading_relative = flat_obs_array[index_map['heading_relative']]
        pos = infos[0].get('proj_dist_from_start', np.array([np.nan]))
        # -----------------------------------------------------------------
        
        # --- Verification Step ---
        angle_in_radians = np.deg2rad(current_course_relative)
        calculated_cmg = current_ground_speed * np.cos(angle_in_radians)

        # new: ready for recording action for each step
        current_action = action[0]

        # Record data for CSV
        data_records.append({
            "step": step_ct,
            "proj_dist": pos,
            "pos_x": pos_x, # new
            "pos_y": pos_y, # new
            "cmg_env": current_cmg,
            "cmg_calc": calculated_cmg,
            "ground_speed": current_ground_speed,
            "course_relative": current_course_relative,
            "heading_relative": current_heading_relative,
            "reward": reward[0], # Reward is also in a list
            "action": current_action # new
        })

        # Print step info
        #print(f"Step {step_ct:2}: CMG={current_cmg:5.2f} (Calc: {calculated_cmg:5.2f}) | "
        #      f"G.Speed={current_ground_speed:5.2f} | C.Err={current_course_relative:5.2f}°")

        print(step_ct,current_cmg)

        if done[0]:
            print(f"\n--- Episode 1 Finished ---")
            print(f"Total steps: {step_ct}")
            if "TimeLimit.truncated" in infos[0]:
                print("Reason: Episode Truncated (reached max step limit).")
            else:
                print("Reason: Episode Terminated (e.g., boat went out of bounds).")
            break

except Exception as e:
    print(f"\nError during simulation loop at step {step_ct}: {e}")
finally:
    print("\nSimulation loop finished. Closing environment.")
    env.close()

# --- Save Data to CSV ---
print(f"\nSaving results to '{OUTPUT_CSV}'...")
try:
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved successfully: {OUTPUT_CSV}")
except Exception as e:
    print(f"Error saving CSV: {e}")

print("\nTest complete. Data ready for plotting.")
