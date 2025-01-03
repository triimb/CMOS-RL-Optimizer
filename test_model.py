import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env
from stable_baselines3 import A2C
from environment import CmosInverterEnvironmentDiscrete
from utils import TensorboardCallback

import logging
logging.basicConfig(level=logging.INFO)


def train_model(env, total_timesteps=1000, log_dir="./logs/"):
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model.learn(total_timesteps=total_timesteps, callback=TensorboardCallback())
    return model

# Extract the best case (W, L)
def extract_best_case(env):
    # TODO
    ...


if __name__ == "__main__":
    
    env = CmosInverterEnvironmentDiscrete()
    
    model = train_model(env, total_timesteps=10)

    # Extract the best design parameters (W and L)
    #best_design = extract_best_case(env)
    #print("Best Design Parameters (W, L):", best_design)
    # TO DO Run simulation for these best parameters
