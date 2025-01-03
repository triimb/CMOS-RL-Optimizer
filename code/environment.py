from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from utils import (
    read_wrdata_file,
    run_exe,
    NGSpiceEnvironment,
    parse_spice_float
)


class CmosInverterEnvironment(NGSpiceEnvironment):
    r"""
    Custom gym environment for optimizing a CMos inverter.

    Private attributes starts with _, those attributes must be kept
    as is after their definition in the constructor __init__.

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
        inverter_path = Path(netlist) # TODO : Path to netlist
        if not inverter_path.exists():
            raise NotImplementedError("Path to netlist is missing")
        super().__init__(inverter_path)
        


        # TODO : Setup all useful class attributes you need in your functions
        self.widths = widths
        self.lengths = lengths
        self.borders = borders or {}
        self.data = None
        
        # TODO : Filter parameters
        #        By default, all .param in the netlists are considered
        #        as parameters, however all parameters must be defined
        #        in the actions, which is not necessarily relevant
        #        (for example a vdd parameter with the voltage power
        #        value).
        #        ⠀
        #        These .param can be removed from the netlist, or kept
        #        (because for example you use it in the code to get
        #        some meta values that must not be optimized).
        #        ⠀
        #        If you keep it, you must separate the parameters in
        #        two dictionaries : self._parameters and
        #        self._hidden_parameters. The second one must exist in
        #        all cases, if you don't have hidden parameters set it
        #        to empty : self._hidden_parameters = {}
        
        self._hidden_parameters = {
            key: value for key, value in self._parameters.items()
            if not key.startswith("W")
        }
        
        if widths:
            self._parameters = {key: value for key, value in self._parameters.items() if key.startswith("W")}
        if lengths:
            self._parameters.update({key: value for key, value in self._parameters.items() if key.startswith("L")})

        self._parameters.update(self.borders)
        


        # TODO : Define the action space : self._action_space
        #        Use gym spaces for that, remember that the
        #        actions are dictionaries associating
        #        parameters to their new values (don't forget
        #        the limits)
        self.action_space = gym.spaces.Dict({
            'W_P': gym.spaces.Box(low=1e-6, high=10e-6, shape=(), dtype=np.float32),  # width range for PMOS
            'W_N': gym.spaces.Box(low=1e-6, high=10e-6, shape=(), dtype=np.float32),  # width range for NMOS
        })
        
        # TODO : define the observation space : self.observation_space
        #        As for action space, use gym spaces to define it, the
        #        observations are a dictionary associating each metric
        #        to its value
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
        None
        """
        # Generate netlist

        # Run simulation and extract data
        # TODO : Write a code to call the simulator and to extract
        #        data from output files.
        #        Some class attributes can be helpful :
        #          - self._netlist : contains path to the netlist to
        #            simulate, netlist has been parsed with the
        #            parameters value given by action dictionary
        #          - self._generated_files : list of files generated
        #            by the simulation (wrdata instructions) (only
        #            the name of the file, not their path)
        #        Also you can use the following class method from
        #        parent :
        #          - get_output_path
        #        You have the following utils functions to help you :
        #          - utils.run_exe
        #          - utils.read_wrdata_file
        #        ⠀
        #        Observations must be a dictionary {
        #           "metric_1": 0.1,
        #           "metric_2": 0.2,
        #        }, I recommend you to write other methods _compute_*
        #        and call it to construct this dictionary : {
        #           "metric_1": compute_metric_1(data),
        #           "metric_2": compute_metric_2(data),
        #           ...
        #        }
            
        
        # Génération du fichier netlist
        with open(self._netlist, "w") as fd:
            fd.write(self._netlist_content.format(**self._parameters, **self._hidden_parameters))
        
        run_exe("ngspice", "--batch", self._netlist)

        self.data = {filename: read_wrdata_file(self.get_output_path(filename)) for filename in self._generated_files}
        #print(f"Il y a {len(data)} fichier générés")
        
        # Plot (optionnal)
        for df in self.data.values():
            #print(df)
            #self.plot_voltages(df)    
            return {
                "power": self._compute_power(df), 
                "delay": self._compute_delay(df),
                "surface": self._compute_surface(df)
            }      
            

        raise NotImplementedError("NGSpiceEnvironment._get_ops method not implemented")


    def _get_info(self):
        return {}

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
        if df is not None:
            vout = df['v(out)'].values
            iout = df['i(vdd)'].values

            return np.mean(vout) * np.mean(iout)
        else:
            raise NotImplementedError("DataFrame Empy")

    def _compute_delay(self, df: dict) -> float:
        if df is not None:
            VDD = 3.3
                
            low_threshold = 0.1 * VDD  # 10% of Vdd
            high_threshold = 0.9 * VDD  # 90% of Vdd
            
            # Step 1: Calculate tPLH (Low-to-High Propagation Delay)
            # Find the time when the output voltage crosses 10% (low) and 90% (high)
            low_to_high_start = df[df['v(out)'] >= low_threshold].iloc[0]
            low_to_high_end = df[df['v(out)'] >= high_threshold].iloc[0]
            
            tPLH = low_to_high_end['time'] - low_to_high_start['time']
            
            # Step 2: Calculate tPHL (High-to-Low Propagation Delay)
            # Find the time when the output voltage crosses 90% (high) and 10% (low)
            high_to_low_start = df[df['v(out)'] >= high_threshold].iloc[0]
            high_to_low_end = df[df['v(out)'] <= low_threshold].iloc[0]
            
            tPHL = high_to_low_end['time'] - high_to_low_start['time']
            
            return (tPLH + tPHL) / 2

        else:
            raise NotImplementedError("DataFrame Empy")
        
    
    def _compute_reward(self, obs: dict) -> float:
        r"""
        Compute the reward.

        Returns
        -------
        Float
            The reward, as a float, or anything you want (but be careful).
        """
        # TODO : Write this method and return the reward, you are
        #        free to replace *args by any number of arguments you
        #        need
        # Raw observations
        power = obs.get("power", 0.0)
        delay = obs.get("delay", 0.0)
        surface = obs.get("surface", 0.0)
        
        #print(f"Power : {power}")
        #print(f"Delay : {delay}")
        #print(f"Surface : {surface}")

        power_min, power_max = 0.0, 10e-6
        delay_min, delay_max = 0.0, 10e-10
        surface_min, surface_max = 0.0, 10e-11

        # Normalized values
        power_norm = (power - power_min) / (power_max - power_min)
        delay_norm = (delay - delay_min) / (delay_max - delay_min)
        surface_norm = (surface - surface_min) / (surface_max - surface_min)

        # Hyperparameters
        alpha = 1.0
        beta = 1.0
        gamma = 0.8

        reward = -(alpha * power_norm + gamma * surface_norm) + beta / (delay_norm + 1e-6)
        
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

    def __init__(self):
        super().__init__()

        # Back-end environment
        self._env = CmosInverterEnvironment('code/cmos_inverter.cir', widths=True, lengths=False, borders=None)
        
        self._observations_order = list(self._env.observation_space.keys())

        # TODO : Continuous environment are not the best to use in
        #        reinforcement learning for this task, we go to a
        #        discrete actions environment.
        #        The discrete actions are a step added or subtracted
        #        in the parameter values :
        #            param_1_action == 1 => param_1_value += step
        #        ⠀
        #        Stable baselines 3 does not want dictionary actions,
        #        the workaround is to define a front-end environment
        #        for SB3 that uses the previous environment in
        #        back-end.
        #        ⠀
        #        And the actions are now arrays :
        #            [param_1_action, param_2_action, ...]
        #        ⠀
        #        Write the new action space.
        self.step_size = 1e-6  # Define the step size
        self.action_space = gym.spaces.MultiDiscrete([3] * len(self._env.action_space.keys()))  
        # Each parameter has 3 options: -1 (decrease), 0 (no change), 1 (increase)

        # TODO : The issue with SB3 and Dict spaces is the same for
        #        observation space. Adapt it in the same way as action
        #        space.self._observations_order = list(self._env.observation_space.keys())
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
        # TODO : Write a code that converts the actions from discrete
        #        to direct values, call the step method of the
        #        back-end environment self._env, then convert the
        #        observation to the SB3 valid format before return.
        #        ⠀
        #        Don't forget anything :)
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

            current_value = self._parameters[param_name]
            new_value = np.clip(current_value + delta, param_space.low, param_space.high)
            action_dict[param_name] = new_value

        # Step the backend environment
        obs, reward, terminated, truncated, info = self._env.step(action_dict)

        obs_array = np.array([obs[key] for key in self._observations_order])

        return obs_array, reward, terminated, truncated, {"obs": obs, **info}


    def close(self):
        super().close()
        self._env.close()
