<div align="left" style="position: relative;">
<h1>CMOS-RL-OPTIMIZER</h1>
<p align="left">
	<code>This project aims to automate the synthesis of a simple CMOS inverter circuit using Reinforcement Learning (RL). The goal is to optimize circuit performance in terms of delay, power, and area by adjusting transistor sizes.</code>
</p>
<p align="left">
	<img src="https://img.shields.io/github/last-commit/triimb/CMOS-RL-Optimizer?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/triimb/CMOS-RL-Optimizer?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/triimb/CMOS-RL-Optimizer?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="left"><!-- default option, no dependency badges. -->
</p>
<p align="left">
	<!-- default option, no dependency badges. -->
</p>
</div>
<br clear="right">

##  Table of Contents

- [ Structure](#-project-structure)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
- [ License](#-license)

##  Project Structure

```sh
└── CMOS-RL-Optimizer/
    ├── bin
    │   └── ngspice
    ├── code
    │   ├── environment.py
    │   └── utils.py
    ├── models
    │   ├── 45nm_bulk.txt
    │   └── AMS035.txt
    ├── netlists
    │   └── cmos_inverter.cir
    ├── tensorboard_logs
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── main.py
```


##  Getting Started

###  Installation

Install CMOS-RL-Optimizer using this method :

1. Clone the CMOS-RL-Optimizer repository:
```sh
❯ git clone https://github.com/triimb/CMOS-RL-Optimizer
```

2. Navigate to the project directory:
```sh
❯ cd CMOS-RL-Optimizer
```

3. Create and activate the virtual environment:

```sh
❯ python -m venv venv
❯ source venv/bin/activate
```

4. Install the dependencies:

```sh
❯ pip install -r requirements.txt
```

5. Navigate to the `bin` folder to make the `ngspice` executable runnable:

```sh
❯ cd bin
❯ chmod +x bin/ngspice
```




###  Usage
Run CMOS-RL-Optimizer using `main.py`:

```sh
❯ python3 main.py
```

## Functions in `main.py`

There are two functions in `main.py`:

1. **`train_model()`**: This function is used to train the environment.
2. **`run_simulation()`**: This function is used to visualize the simulation results with the optimal parameters.

The optimal parameters should be retrieved from TensorBoard, located in the `tensorboard_logs` folder. Then, these parameters should be added as a `.param` file in the netlist located at `netlists/cmos_inverter.cir`.

*Optimal parameters are already set, so you can simply comment out the `train_model()` function and run `run_simulation()` for now (or check the parameters in TensorBoard).*

##  License

This project is protected under the [MIT](https://choosealicense.com/licenses/mit/#) License.
