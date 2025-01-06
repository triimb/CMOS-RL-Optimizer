from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from code.utils import (
    read_wrdata_file,
    run_exe,
    NGSpiceEnvironment,
    parse_spice_float
)


class CmosInverterEnvironment(NGSpiceEnvironment):
    r"""
    Custom gym environment for optimizing a CMOS inverter.

    Attributes
    ----------
    _netlist_src : Path
        Path to original netlist, use at beginning to load the netlist.
    _tmpdir : Path
        Tempory dictionary that contains modified netlists to run.
    _netlist_content : str
        Unformatted string representing the content of the simulation
        netlist.
    _netlist : Path
        Path to the ouput file for the formatted netlist.
    """

    def __init__(
            self,
            netlist,
            widths: bool = True,
            lengths: bool = False,
            borders: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize the CMOS inverter environment.

        Parameters
        ----------
        netlist : str
            Path to the netlist file.
        widths : bool, optional
            Whether to optimize transistor widths (default is True).
        lengths : bool, optional
            Whether to optimize transistor lengths (default is False).
        borders : dict, optional
            Parameter value boundaries, e.g., {'W_P': (1e-6, 10e-6)}.
        """

        inverter_path = Path(netlist)
        if not inverter_path.exists():
            raise NotImplementedError("Path to netlist is missing")
        super().__init__(inverter_path)
        
        # Class attributes for environment configuration
        self.widths = widths
        self.lengths = lengths
        self.borders = borders or {}
        self.data = None
        
        # Filter parameters
        # - Hidden parameters: not exposed for optimization (e.g., supply voltage)
        # - Actionable parameters: e.g., widths starting with "W" or lengths "L"
        self._hidden_parameters = {
            key: value for key, value in self._parameters.items() if not key.startswith("W")
        }
        
        if widths:
            self._parameters = {key: value for key, value in self._parameters.items() if key.startswith("W")}
        if lengths:
            self._parameters.update({key: value for key, value in self._parameters.items() if key.startswith("L")})

        self._parameters.update(self.borders)

        # Define action space
        self.action_space = gym.spaces.Dict({
            'W_P': gym.spaces.Box(low=1e-6, high=10e-6, shape=(), dtype=np.float32),  # PMOS width range
            'W_N': gym.spaces.Box(low=1e-6, high=10e-6, shape=(), dtype=np.float32),  # NMOS width range
        })
        
        # Define observation space
        self.observation_space = gym.spaces.Dict({
            "power": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            "delay": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),
            "surface": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32)
        })


    def _get_obs(self) -> dict:
        r"""
        Run the simulation and extract the data.

        Returns
        -------
        dict
            Observations, e.g., {"power": 0.5, "delay": 1.2, "surface": 2.3}.
        """
        
        # Generate netlist with updated parameters
        with open(self._netlist, "w") as fd:
            fd.write(self._netlist_content.format(**self._parameters, **self._hidden_parameters))
        
        # Run the simulator
        run_exe("./bin/ngspice", "--batch", self._netlist)

        # Read first output file
        self.data = read_wrdata_file(self.get_output_path(self._generated_files[0]))
        
        if self.data is not None:
            return {
                "power": self._compute_power(self.data), 
                "delay": self._compute_delay(self.data),
                "surface": self._compute_surface(self.data)
            }      
        else:
            raise NotImplementedError("DataFrame Empy")


    def _get_info(self):
        return {
            "W": {key: self._parameters[key] for key in self._parameters if key.startswith("W")},
            "L": {key: self._parameters[key] for key in self._parameters if key.startswith("L")},
            "Surface": self._compute_surface(self.data)
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        r"""
        Reset the environment.

        Parameters
        ----------
        seed : seed, optional
            Seed to use for reproducibility.
        options : dict, optional
            Options for the environment, unused.

        Returns
        -------
        observation : dict
            Dictionary of observations, for example : {
                "metric_1": 0.5,
                "metric_2": 1.2,
            }
        info : dict
            Dictionary with useful information, empty in this
            environment.
        """
        super().reset(seed=seed)

        # Sample new parameter values from the action space
        params = self.action_space.sample()
        self._parameters.update(params)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def plot_voltages(self, df : pd.DataFrame):
        """
        Plots v(in) and v(out) vs time.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing simulation results with 'time', 'v(in)', and 'v(out)'.
        """
        
        if df is not None:
            time = df['time'].values
            vin = df['v(in)'].values
            vout = df['v(out)'].values
        
        plt.figure(figsize=(10, 6))
        plt.plot(time, vin, label='v(in)', color='blue')
        plt.plot(time, vout, label='v(out)', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Input and Output Voltage vs Time')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()
        
    def _compute_surface(self, df: dict) -> float:
        r"""
        Compute the transistor surface area.
        
        Surface = W_N * L_N + W_P * L_P

        Returns
        -------
        float
            Total surface area, e.g., 2e-12.
        """
        if self.lengths:
            return(
                self._parameters["W_N"] * self._parameters["L_N"] +
                self._parameters["W_P"] * self._parameters["L_P"]
            )
        else:
            return(
                self._parameters["W_N"] * self._hidden_parameters["L_N"] +
                self._parameters["W_P"] * self._hidden_parameters["L_P"]
            )
            
    
    def _compute_power(self, df: dict) -> float:
        """
        Compute average power consumption.
        
        Power = v(out) * i(vdd)

        Returns
        -------
        float
            Power consumption, e.g., 5.2e-6 W
        """
        if df is not None:
            vout = df['v(out)'].values
            iout = df['i(vdd)'].values
            return abs(np.mean(vout) * np.mean(iout))
        else:
            raise NotImplementedError("DataFrame Empty")

    def _compute_delay(self, df: dict) -> float:
        """
        Compute propagation delay.

        t_p = (tPLH + tPHL) / 2

        Returns
        -------
        float
            Average propagation delay, e.g., 1.5e-9.
        """
        if df is not None:
            VDD = 3.3
                
            low_threshold = 0.1 * VDD  # 10% of VDD
            high_threshold = 0.9 * VDD  # 90% of VDD
            
            # Find the time when the output voltage crosses 10% (low) and 90% (high)
            low_to_high_start = df[df['v(out)'] >= low_threshold].iloc[0]
            low_to_high_end = df[df['v(out)'] >= high_threshold].iloc[0]
            tPLH = low_to_high_end['time'] - low_to_high_start['time']
            
            # Find the time when the output voltage crosses 90% (high) and 10% (low)
            high_to_low_start = df[df['v(out)'] >= high_threshold].iloc[0]
            high_to_low_end = df[df['v(out)'] <= low_threshold].iloc[0]
            tPHL = high_to_low_end['time'] - high_to_low_start['time']
            
            return (tPLH + tPHL) / 2

        else:
            raise NotImplementedError("DataFrame Empty")
        
    
    def _compute_reward(self, obs: dict) -> float:
        r"""
        Compute the reward based on the observations.

        Parameters
        ----------
        obs : dict
            A dictionary containing the observation values. Expected keys are:
            - "power": float, represents the power value.
            - "delay": float, represents the delay value.
            - "surface": float, represents the surface value.

        Returns
        -------
        float
            The computed reward, based on normalized values of power, delay, and surface.
            The reward function is designed to penalize high power and surface values,
            and reward lower delay values.
        """

        # Raw observations
        power = max(obs.get("power", 0.0), 1e-10)
        delay = obs.get("delay", 0.0)
        surface = obs.get("surface", 0.0)

        power = max(power, 1e-10)  # Avoid negative or zero power

        # Normalization bounds (adjust if necessary)
        power_min, power_max = 1e-10, 5e-5  
        delay_min, delay_max = 1e-12, 1e-9  
        surface_min, surface_max = 1e-15, 5e-9

        # Normalize the observation values
        power_norm = np.log10(power) / np.log10(power_max)  
        delay_norm = (delay - delay_min) / (delay_max - delay_min)
        surface_norm = np.log10(surface) / np.log10(surface_max)

        # Hyperparameters to weight different components of the reward
        alpha = 0.3
        beta = 0.4
        gamma = 0.3

        reward = -(alpha * power_norm + gamma * surface_norm) + beta / (delay_norm + 1e-6)

        if power > power_max * 1.5:
            reward -= 0.2  # penalty if power consumption ++
        if delay > delay_max * 1.5:
            reward -= 0.2  # penalty if delay ++
        if surface > surface_max * 1.5:
            reward -= 0.2  # penalty if surface area ++

        
        return reward
        
    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        r"""
        Do one step in the environment.

        Parameters
        ----------
        action : dict
            Dictionary of the new values for the parameters, for
            example : {
                "param_1": 0.1,
                "param_2": 0.2,
            }
            Parameters in the dictionary must match the .param
            parameters in the netlist.

        Returns
        -------
        observation : dict
            Dictionary of observations, for example : {
                "metric_1": 0.5,
                "metric_2": 1.2,
            }
        reward : float
            Floating value representing the reward.
        terminated : bool
            If the environment as reach the end, always False because
            non-episodic environment.
        truncated : bool
            If the environment ended unexpectedly, always False
            because the environment cannot stop.
        info : dict
            Dictionary with useful information, empty in this
            environment.
        """
        self._parameters.update(action)

        obs = self._get_obs()

        # TODO : Pass the arguments needed to compute the reward
        reward = self._compute_reward(obs)

        info = self._get_info()

        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info


class CmosInverterEnvironmentDiscrete(gym.Env):
    """
    A discrete action-space wrapper for the CmosInverter environment, designed
    to work with stable baselines 3 (SB3) by converting continuous action and observation spaces
    to discrete representations.
    """

    def __init__(
            self,
            netlist,
            widths: bool = True,
            lengths: bool = False,
            borders: Optional[dict[str, tuple[float, float]]] = None,
        ):
        super().__init__()

        # Back-end environment
        self._env = CmosInverterEnvironment(netlist, widths, lengths, borders)
        
        # Observation space from the back-end environment
        self._observations_order = list(self._env.observation_space.keys())

        # Discrete action space: each parameter has 3 possible actions (-1, 0, 1)
        self.step_size = 1e-6  # Step size for parameter changes
        self.action_space = gym.spaces.MultiDiscrete([3] * len(self._env.action_space.keys()))  
        # Each parameter has 3 possible actions: -1 (decrease), 0 (no change), 1 (increase)

        # Observation space (converted from back-end environment)
        obs_low = np.array([self._env.observation_space[key].low for key in self._observations_order])
        obs_high = np.array([self._env.observation_space[key].high for key in self._observations_order])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    @property
    def _parameters(self):
        return self._env._parameters

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self._env.reset()

        return (
            np.array([obs[key] for key in self._observations_order]),
            {"obs": obs, **info},
        )

    def step(self, action):
        """
        Takes a step in the environment with the provided discrete action.

        Parameters
        ----------
        action : array-like
            The discrete action values to apply to the environment. An array of integers.
        
        Returns
        -------
        tuple
            A tuple containing:
            - An array of the next observations, in SB3-compatible format.
            - A float representing the reward.
            - A boolean indicating whether the environment has terminated.
            - A boolean indicating whether the environment has truncated.
            - A dictionary containing additional information.
        """
        action_dict = {}
        for idx, (param_name, param_space) in enumerate(self._env.action_space.items()):
            discrete_action = action[idx]
            if discrete_action == 0:  # No change
                delta = 0
            elif discrete_action == 1:  # Increase
                delta = self.step_size
            elif discrete_action == 2:  # Decrease
                delta = -self.step_size
            else:
                raise ValueError(f"Unexpected discrete action value: {discrete_action}")

            # Get the current value of the parameter, apply the delta, and ensure it stays within bounds
            current_value = self._parameters[param_name]
            new_value = np.clip(current_value + delta, param_space.low, param_space.high)
            action_dict[param_name] = new_value

        # Step the backend environment
        obs, reward, terminated, truncated, info = self._env.step(action_dict)

        # Convert observations to SB3-compatible format
        obs_array = np.array([obs[key] for key in self._observations_order])

        return obs_array, reward, terminated, truncated, {"obs": obs, **info}


    def close(self):
        super().close()
        self._env.close()
