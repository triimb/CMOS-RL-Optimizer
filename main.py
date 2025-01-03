import random
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from math import ceil, log10

from utils import TensorboardCallback
from environment import CmosInverterEnvironment
from environment import CmosInverterEnvironmentDiscrete

def make_env(env_id, rank, seed=0):
    def _init():
        env = CmosInverterEnvironmentDiscrete()
        env.reset(seed=seed + rank) 
        return env
    return _init


if __name__ == "__main__":
    num_envs = 4 

    envs = SubprocVecEnv([make_env("CmosInverterEnvironmentDiscrete", i) for i in range(num_envs)])

    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = A2C("MlpPolicy", envs, verbose=1, tensorboard_log="./a2c_tensorboard/")

    model.learn(total_timesteps=100, callback=TensorboardCallback())

    model.save("a2c_cmos_inverter")

    envs.close()    
        
