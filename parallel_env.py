import multiprocessing as mp
from typing import Callable, List, Optional, Any, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper, VecEnvIndices, VecEnvStepReturn
)

# --- Worker Function ---
def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env_fn_wrapper: CloudpickleWrapper) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                if terminated or truncated:
                    info["final_observation"] = observation
                    observation, info_reset = env.reset()
                    info.update(info_reset)
                remote.send((observation, reward, terminated, truncated, info))
            elif cmd == "reset":
                observation, info = env.reset()
                remote.send((observation, info))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break

# --- Custom Vectorizer Class ---
class NonDaemonicSubprocVecEnv(VecEnv):
    """
    A custom SubprocVecEnv that spawns non-daemonic processes.
    Crucial for environments (like PyBullet) that need to spawn their own internal processes.
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # CRITICAL FIX: daemon=False allows child processes to spawn children
            process = ctx.Process(target=_worker, args=args, daemon=False)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, terminated, truncated, infos = zip(*results)
        dones = np.array(terminated) | np.array(truncated)
        return np.stack(obs), np.array(rews), dones, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs)

    def close(self) -> None:
        if self.closed: return
        if self.waiting:
            for remote in self.remotes: remote.recv()
        for remote in self.remotes: remote.send(("close", None))
        for process in self.processes: process.join()
        self.closed = True

    # --- IMPLEMENTED METHODS (Fixes IndexError) ---
    
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environments (e.g. render_mode)"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Union[type, str], indices: VecEnvIndices = None) -> List[bool]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]
        
    def _get_target_remotes(self, indices: VecEnvIndices) -> List[mp.connection.Connection]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]