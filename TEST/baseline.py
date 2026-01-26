import json
from pathlib import Path
import numpy as np
import pandas as pd
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne
from boatsimulator.core.gl.contextmanager import ContextManager

MODEL_DIR = Path("Output")
MODEL_DIR.mkdir(parents=True, exist_ok=True)  

NEUTRAL_ACTION = 1


with open("test.json") as f:
    params = json.load(f)

cm = ContextManager(headless=True)
env = SailboatEnv_consigne("Test env.", params, cm=cm)


OUTPUT_CSV = f"baseline.csv"


data_records = []
obs, info = env.reset()
step_ct = 0

while True:
    step_ct += 1
    obs, reward, terminated, truncated, infos = env.step(NEUTRAL_ACTION)

    # new: ready for tracking current position
    current_pos = infos.get('current_pos', [0, 0])
    pos_x = current_pos[0]
    pos_y = current_pos[1]

    cmg = obs.get('cmg', np.nan)
    crs = obs.get('course_relative', np.nan)
    hrs = obs.get('heading_relative', np.nan)
    spd = obs.get('ground_speed', np.nan)
    pos = infos.get('proj_dist_from_start', np.array([np.nan]))
    cmg_calc = spd * np.cos(np.deg2rad(crs))

    print(step_ct,cmg)

    data_records.append({
        "step": step_ct,
        "proj_dist": pos,
        "pos_x": pos_x, # new
        "pos_y": pos_y, # new
        "cmg_env": cmg,
        "cmg_calc": cmg_calc,
        "ground_speed": spd,
        "course_relative": crs,
        "heading_relative": hrs,
        "reward": reward,
        "action": NEUTRAL_ACTION # new
    })

    if terminated or truncated:
        break

env.close()
pd.DataFrame(data_records).to_csv(OUTPUT_CSV, index=False)
