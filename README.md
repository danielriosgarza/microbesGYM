---

# MicrobesGYM

**Model-based simulation and reinforcement learning framework for microbiome control**

MicrobesGYM combines **kinetic modeling** and **reinforcement learning (RL)** to design, simulate, and train controllers for microbial ecosystems.
It integrates dynamic biochemical modeling with data-driven control, enabling *in silico* experimentation and AI-driven discovery of control strategies.

---

## Overview

MicrobesGYM consists of two complementary components:

* **`kinetic_model`** — a fast and flexible simulation engine for microbiome–metabolome systems.
  It supports pulse-based bioreactor experiments with environmental factors such as pH, temperature, and stirring, offering *fast*, *balanced*, and *accurate* integration modes with interactive visualization.

* **`rl`** — a configurable, model-agnostic reinforcement learning framework that wraps any kinetic model as a Gymnasium environment.
  Targets, observations, rewards, and actions are defined through a single YAML/JSON configuration, and agents are trained via **Stable-Baselines3 (SB3)**.

Together, these modules provide a complete environment for developing adaptive control strategies for microbiomes, from simulation to RL training.

---

## Key Features

* **Config-driven RL design** — define all aspects (actions, observations, rewards, and episode settings) in one declarative file.
* **Model-agnostic interface** — plug in any compatible kinetic model JSON without code modification.
* **Safe expression-based rewards** — flexible reward definitions through parsed mathematical expressions.
* **Performance control** — choose between *fast*, *balanced*, and *accurate* simulation modes.
* **Visualization tools** — interactive Plotly plots and analysis utilities for model outputs.
* **Reproducible workflows** — consistent YAML-based configuration and environment management.

---

## Installation

```bash
git clone https://github.com/<your-username>/microbesGYM.git
cd microbesGYM
pip install .
```

Optional extras:

```bash
pip install ".[extras]"   # tensorboard, tqdm, visualization
pip install ".[all]"      # dev + extras
```

Requires **Python ≥ 3.9**.

---

## Project Structure

```text
microbesGYM/
├─ modelTemplates/         # ready-to-use kinetic model JSONs
├─ notebooks/              # demos and tutorials
├─ src/
│  ├─ kinetic_model/       # simulation engine
│  └─ rl/                  # reinforcement learning framework
├─ trained_models/         # example RL checkpoints
└─ pyproject.toml
```

---

## Quickstart

### 1. Simulate a microbiome system

```python
from kinetic_model import Metabolite, Metabolome, Microbiome, Environment, pH, Temperature, Stirring, Pulse, Reactor

glucose = Metabolite("glucose", 10.0, {"C":6,"H":12,"O":6}, "#ff0000")
metabolome = Metabolome([glucose])
microbiome = Microbiome(name="demo", bacteria={})
env = Environment(pH(metabolome, intercept=7.0), Stirring(rate=0.9), Temperature(37.0))

pulse = Pulse(t_start=0.0, t_end=10.0, n_steps=100, environment=env)
reactor = Reactor(microbiome, metabolome, pulses=[pulse], volume=1.0)
reactor.set_balanced_simulation_mode()
reactor.simulate()
reactor.make_plot()
```

Or load a predefined model:

```python
from kinetic_model import ModelFromJson
model = ModelFromJson("modelTemplates/bh_bt_ri_complete_model_export.json")
```

---

### 2. Train a controller with RL

```bash
python -m rl.cli.train_agent \
  --algo sac \
  --model-json modelTemplates/bh_bt_ri_complete_model_export.json \
  --config src/rl/examples/configs/butyrate_control.yaml \
  --timesteps 100000 \
  --n-envs 4 \
  --eval-freq 5000 \
  --log-dir logs/rl \
  --model-path models/mg_rl_general_sac
```

Evaluate the trained agent:

```bash
python -m rl.cli.evaluate \
  --algo sac \
  --model-json modelTemplates/bh_bt_ri_complete_model_export.json \
  --config src/rl/examples/configs/butyrate_control.yaml \
  --model-path models/mg_rl_general_sac.zip
```

Evaluation automatically loads normalization statistics (if available).

---

## RL Configuration Schema

RL behavior is governed by `TopLevelConfig` (`src/rl/config.py`), expressed in YAML or JSON.
Below is a compact example:

```yaml
episode:
  horizon: 250
  dt_hours: 1.0
  training_mode: balanced
simulation:
  min_steps_per_pulse: 10
  steps_per_hour_factor: 50
actions:
  actuators:
    - { name: q, type: continuous }
    - { name: temp, type: continuous }
observations:
  include: ["met.all", "pH.used"]
  pipeline:
    - { normalize: { method: running_mean_var } }
rewards:
  terms:
    - { expr: delta_target, weight: 200.0, deadband: 0.02 }
target:
  type: metabolite
  name: butyrate
  use_delta: true
```

See detailed documentation in `src/rl/docs/DOCUMENTATION.md`.

---

## Examples and Notebooks

* **`notebooks/kinetic_model.ipynb`** — microbiome modeling and simulation
* **`notebooks/controller_simulation_demo.ipynb`** — RL-based control demo
* **`src/rl/examples/configs/`** — pre-tuned control configs (butyrate, acetate, succinate, kombucha, etc.)
* **`modelTemplates/`** — curated kinetic models for simulations

---

## Command Line Interface

* `python -m rl.cli.train_agent --help`
* `python -m rl.cli.evaluate --help`
* `python -m kinetic_model.cli`

Supports SAC, PPO (and TQC via `sb3-contrib`), TensorBoard logging, checkpointing, and normalization syncing between environments.

---

## Citation

If you use **MicrobesGYM** in your research, please cite the GitHub repository:

```
@misc{MicrobesGYM2025,
  author       = {Rios Garza, Daniel},
  title        = {MicrobesGYM: Kinetic Modeling and Reinforcement Learning Framework for Microbiome Control},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/microbesGYM}},
  note         = {Accessed October 2025}
}
```

A formal BibTeX entry will be updated once a preprint is available.

---

## Contributing

Contributions are welcome!
For major changes, please open an issue first to discuss your proposal.
Add minimal tests for new features, and keep code typed and readable.

---

## License

MIT License © 2025 [Daniel Rios Garza](https://github.com/<your-username>)


## Frontend
A `React` app will soon be lauched and announced here. Checkout our poster.
---

