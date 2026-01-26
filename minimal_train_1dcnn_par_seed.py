import json
import random
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable

# Gymnasium & SB3
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize
from stable_baselines3.common.utils import set_random_seed


from stable_baselines3.common.callbacks import CheckpointCallback 

#sss

# Custom env / policy
from boatsgym.envs.consigne.sailboat_consigne import SailboatEnv_consigne
from boatsimulator.core.gl.contextmanager import ContextManager
from cnn_extractor import HistoryCNNExtractor
from parallel_env import NonDaemonicSubprocVecEnv


from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

'''
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: The initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining) :
        # progress_remaining starts at 1.0 (beginning) and goes to 0.0 (end)
        return progress_remaining * initial_value
    return func
'''
def linear_schedule(initial_value: float, final_value: float = 0.0) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    
    :param initial_value: The initial learning rate.
    :param final_value: The final learning rate.
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining starts at 1.0 (start) and linearly decays to 0.0 (end)
        # Formula: end + progress * (start - end)
        return final_value + progress_remaining * (initial_value - final_value)
    
    return func

# -------------------- Progress Bar --------------------
class ProgressBarCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self._pbar = pbar

    def _on_step(self) -> bool:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True


# -------------------- Seeding --------------------
def set_all_seeds(seed: int):
    set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# -------------------- Env factory --------------------
def make_env(params):
    def _init():
        cm = ContextManager(headless=True)
        env = SailboatEnv_consigne("Test env.", params, cm=cm)
        env = Monitor(env)
        env = FlattenObservation(env)
        #env.reset(seed=seed)
        return env
    return _init



class SyncVecNormalizeCallback(BaseCallback):
    """
    Callback that synchronizes the normalization statistics (obs_rms)
    from the training environment to the evaluation environment.
    
    Without this, the evaluation environment would try to normalize data
    using default values (mean=0, var=1), leading to incorrect inputs 
    for the agent during testing.
    """
    def __init__(self, train_env, eval_env):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        # Copy the Running Mean/Std of observations from Train to Eval
        # We generally do NOT sync reward stats (ret_rms) because 
        # evaluation rewards should be raw/unnormalized for human readability.
        self.eval_env.obs_rms = self.train_env.obs_rms
        return True
    



# -------------------- Main --------------------
if __name__ == "__main__":

    # ===== Experiment config =====
    TRAIN_STEPS = 250000  # 100000
    N_PROCS = 5
    N_STACK = 4
    #SEEDS = [0, 1, 2]
    SEEDS = [0,1]

    BASE_DIR = Path("model") / f"Par_{TRAIN_STEPS}_1DCNN"
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    with open("train.json") as f:
        params = json.load(f)

    # ===== Run multiple seeds =====
    for seed in SEEDS:
        print(f"\n==============================")
        print(f" Running seed {seed}")
        print(f"==============================")

        run_dir = BASE_DIR / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- Set all RNGs ---
        set_all_seeds(seed)

        # 1. Create Training Environment
        # SB3 will automatically assign seeds: seed, seed+1, seed+2... to the workers
        train_env = make_vec_env(
            make_env(params),
            n_envs=N_PROCS,
            seed=seed,
            vec_env_cls=NonDaemonicSubprocVecEnv,
        )
        train_env = VecFrameStack(train_env, n_stack=N_STACK, channels_order="first")
        # Training=True: We LEARN the stats (mean/var) here
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

        # 2. Create Evaluation Environment ### IMPORTANT --> EVAL LOGIC IS REMOVED SINCE WE ARE USING CUSTOM CHEACK
        #POINTS
        # Use a distinct seed offset (e.g., +1000) so validation weather/targets are unique
        #eval_env = make_vec_env(
        #     make_env(params), 
        #    n_envs=1, 
        #    seed=seed + 1000
        #)
        
        #eval_env = VecFrameStack(eval_env, n_stack=N_STACK, channels_order="first")
        # Training=False: We USE stats here, but do not update them
        #eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.)

        # 3. Configure Callbacks
        
        # A. Sync Stats: Copies "what is normal wind" from Train -> Eval
        #sync_cb = SyncVecNormalizeCallback(train_env, eval_env)

        # B. Eval: Tests the agent periodically
        checkpoint_cb = CheckpointCallback(
            save_freq= 1000,             # Save every 10,000 timesteps
            save_path=str(run_dir / "cheakpoints"),
            name_prefix="ppo_sailboat",        
            save_vecnormalize=True        # CRITICAL: Saves the environment stats (mean/std)
        )

        '''
        # for eval *5  is not enough (5 --> 100, eval during traning is too slow)--->  do the  full traning as usual, save the model, 
        # save the model every 100,00 (so instead of one final model )
        # explore the reward
        # HINT - >  checkpoint call back (check baseline)
        # pick the best 
        '''

        # --- Policy config ---
        policy_kwargs = dict(
            features_extractor_class=HistoryCNNExtractor,
            features_extractor_kwargs=dict(
                features_dim=64,
                n_stack=N_STACK,
            ),
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        )
        lr_schedule = linear_schedule(1e-3, 1e-5)


        # --- Model Setup ---
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=lr_schedule, 
            n_steps=128,
            batch_size=128,
            gamma=0.995,        # Higher gamma for longer episodes (look further ahead)
            gae_lambda=0.92, #0.95
            ent_coef=0.01,      # Encourage exploration
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(run_dir)
        )
        
        '''
        Try the following values for the learning rate: 
            10^-2, 5*10^-3, 10^-3
            5*10^-4, 
            10^-4
            And plot the learning curves
        '''


        # --- Train ---
        with tqdm(total=TRAIN_STEPS, desc=f"Seed {seed}") as pbar:
            model.learn(
                total_timesteps=TRAIN_STEPS,
                # List of callbacks: Progress -> Sync -> Eval
                callback=[ProgressBarCallback(pbar), checkpoint_cb], # No EvalCallback
                tb_log_name="tb_1dcnn",
                reset_num_timesteps=True,
            )

        # --- Save Final Model ---
        model.save(run_dir / "final_model_ns128.zip")
        train_env.save(run_dir / "vec_normalize_ns128.pkl")

        # Cleanup
        train_env.close()
        #eval_env.close()

    print("All seeds finished successfully.")
