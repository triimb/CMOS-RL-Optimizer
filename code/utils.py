import abc
import logging
from math import ceil
import os
from pathlib import Path
from pprint import pprint
import shutil
import subprocess
import tempfile
from typing import Any

import gymnasium as gym
from numpy import log10
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


logger = logging.getLogger("utils")


def parse_spice_float(value: str) -> float | int:
    r"""
    Parse a SPICE float value to Python float.

    Be careful : do not use non parsed suffix after the value (for
    example "16uF" will not be parsed correctly, use only "16u").

    Parameters
    ----------
    value : str
        The SPICE float value to parse.

    Returns
    -------
    float or int
        The parsed value.
    """
    value = value.lower()

    if "m" in value:
        return float(value.replace("m", "")) * 1e-3
    elif "u" in value:
        return float(value.replace("u", "")) * 1e-6
    elif "n" in value:
        return float(value.replace("n", "")) * 1e-9
    elif "p" in value:
        return float(value.replace("p", "")) * 1e-12
    elif "f" in value:
        return float(value.replace("f", "")) * 1e-15
    elif "k" in value:
        return float(value.replace("k", "")) * 1e3
    elif "meg" in value:
        return float(value.replace("meg", "")) * 1e6
    elif "g" in value:
        return float(value.replace("g", "")) * 1e9
    elif "t" in value:
        return float(value.replace("t", "")) * 1e12
    else:
        try:
            return int(value)
        except ValueError:
            return float(value)


def run_exe(exe: str | Path, *options: Any) -> None:
    r"""
    Run some system command.

    Command or executable can be passed as a path to the file
    executable location, or only by its name if the executable
    directory is in the PATH environment variable.

    Examples
    --------
    >>> run_exe("/bin/ls", "-l", "-a", "-h", "/home/me")

    Parameters
    ----------
    exe : str | Path
        Path to executable to run.
    options : list
        Options for the command to run as a list.

    Returns
    -------
    None
    """
    args = [exe, *[
        option.as_posix()
        if isinstance(option, Path)
        else str(option)
        for option in options
    ]]

    logger.debug("Running command '%s'", ' '.join(args))

    subprocess.run(
        args,
        check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def read_wrdata_file(
        filepath: str | Path,
        *,
        erase: bool = True,
) -> pd.DataFrame:
    r"""
    Read data generated by wrdata ngspice instruction.

    Parameters
    ----------
    filepath : str or Path
        Path to the file to read.
    erase : bool, default=True
        If the file must be erased after reading, is used to ensure
        a new file will be created at the next simulation.

    Returns
    -------
    pandas.DataFrame
        File content as pandas dataframe.
    """
    data = pd.read_csv(filepath, sep="\s+")

    if erase:
        os.remove(filepath)

    return data


from torch.utils.tensorboard import SummaryWriter
class TensorboardCallback(BaseCallback):
    r"""
    Callback for logging reward, obs and action to Tensorboard.

    If multiple environments are used, the reward, obs and action are logged
    for each environment separately. The keys will be "reward/<env index>",
    "obs/<env index>/<key>", and "actions/<env index>/<key>" respectively.

    Reward information comes from the locals. The obs and action information
    comes from the infos dictionary. They must be "obs" and "action" keys in
    the environment infos dictionary. All the sub-keys of the "obs" and
    "action" must be dictionary associating `key` to value.
    """

    def __init__(self):
        super().__init__()

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        print(f"Rewards: {rewards}")  # Debugging
        print(f"Infos: {infos}")  # Debugging

        if len(rewards) == 1:  # Single environment
            print(f"Logging single environment rewards: {rewards[0]}")
            self.logger.record("reward", rewards[0])

            if "obs" in infos[0]:
                for key, value in infos[0]["obs"].items():
                    print(f"Logging obs/{key}: {value}")  # Debugging
                    self.logger.record(f"obs/{key}", value)

            if "action" in infos[0]:
                for key, value in infos[0]["action"].items():
                    print(f"Logging actions/{key}: {value}")  # Debugging
                    self.logger.record(f"actions/{key}", value)
        else:  # Multiple environments
            digits = ceil(log10(len(rewards) + 1))
            for i, _ in enumerate(rewards):
                i_str = '{num:0{width}d}'.format(num=i, width=digits)
                print(f"Logging multiple environments: reward/{i_str}: {rewards[i]}")  # Debugging
                self.logger.record(f"reward/{i_str}", rewards[i])

                if "obs" in infos[i]:
                    for key, value in infos[i]["obs"].items():
                        print(f"Logging obs/{i_str}/{key}: {value}")  # Debugging
                        self.logger.record(f"obs/{i_str}/{key}", value)

                if "action" in infos[i]:
                    for key, value in infos[i]["action"].items():
                        print(f"Logging actions/{i_str}/{key}: {value}")  # Debugging
                        self.logger.record(f"actions/{i_str}/{key}", value)

        return True



class NGSpiceEnvironment(gym.Env, abc.ABC):
    r"""
    Base class for NGSpice based environments.

    Missing methods to get a fully working gym environment :
        - step(self, action)
        - reset(self)

    Action and observation spaces must also be defined in the
    constructor :
        - self.action_space = ...
        - self.observation_space = ...

    The netlist is editing using the following syntax :
        self._netlist_content.format(**self._parameters)

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

    def __init__(self, unformatted_netlist: str | Path):
        self._netlist_src = Path(unformatted_netlist)

        self._tmpdir = Path(tempfile.mkdtemp())
        (self._tmpdir / "netlists").mkdir()

        with open(self._netlist_src, "r") as fd:
            self._netlist_content = fd.read()

        # Security fix for files edited with Windows
        self._netlist_content.replace("\r\n", "\n")
        # Security fix for legacy MacOS files
        self._netlist_content.replace("\r", "\n")

        # Copy all include files to the temporary directory
        for line in self._netlist_content.split("\n"):
            if line.lower().startswith(".include"):
                include_file = Path(line.split()[1].strip('"'))

                if not include_file.is_absolute():
                    shutil.copy2(self._netlist_src.parent / include_file, self._tmpdir / "netlists" / include_file.name)

        self._netlist = self._tmpdir / "netlists" / self._netlist_src.name

        # Get all parameters and generated files
        self._parameters = {}
        self._generated_files = []

        netlist_content = self._netlist_content.split("\n")

        for i, line in enumerate(netlist_content):
            if line.lower().startswith(".param"):
                params = line.split()[1:]

                line_params = {
                    key: parse_spice_float(value) for key, value in (param.split("=") for param in params)
                }

                self._parameters.update(line_params)

                netlist_content[i] = (
                    line.split()[0] + " "
                    + " ".join(f"{key}={{{key}}}" for key, value in line_params.items())
                )
            if line.lower().strip().startswith("wrdata"):
                self._generated_files.append(line.strip().split()[1])

                netlist_content[i] = line.replace(
                    self._generated_files[-1],
                    (self._tmpdir / self._generated_files[-1]).as_posix()
                )

        self._netlist_content = "\n".join(netlist_content)

        self._closed = False

    @property
    def parameters(self):
        return list(self._parameters.keys())

    def get_output_path(self, filename: str) -> Path:
        r"""
        Get the location of the output file.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        Path
            Path to the file.
        """
        return self._tmpdir / filename

    def __del__(self):
        self.close()

    def render(self) -> None:
        r"""
        Render the environment.

        Print the current parameter values of the environment.

        Returns
        -------
        None
        """
        pprint({
            "parameters": self._parameters,
        })

    def close(self):
        if self._closed is False:
            shutil.rmtree(self._tmpdir)

        self._closed = True