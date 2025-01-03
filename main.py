from code.environment import CmosInverterEnvironmentDiscrete
from code.utils import TensorboardCallback
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

def train_model():
    """
    Train the A2C model using the CMOS inverter environment.
    """
    env = CmosInverterEnvironmentDiscrete(
        netlist='netlists/cmos_inverter.cir',
        widths=True,
        lengths=False,
    )

    # Wrap the environment in a DummyVecEnv for compatibility with Stable Baselines3
    env = DummyVecEnv([lambda: env])

    # Train the model
    model = A2C('MlpPolicy', env, verbose=1, tensorboard_log="tensorboard_logs/")
    model.learn(total_timesteps=20_000, callback=TensorboardCallback()) #SB3 is printing table every 500 steps

    # Save the trained model
    model.save("a2c_cmos_inverter")
    print("Model saved...")

def run_simulation():
    """
    Run a simulation using the trained model and execute an NGSpice simulation.
    """
    from code.utils import run_exe

    run_exe("./bin/ngspice", "netlists/cmos_inverter.cir")

if __name__ == "__main__":
    #train_model()
    run_simulation()
