# microbesGYM

Kinetic modeling and reinforcement learning to build and train microbiome controllers.

microbesGYM brings together two complementary parts:

- `kinetic_model`: a simulation engine to design microbiome/metabolome systems and run pulse-based bioreactor experiments with pH, temperature, and stirring effects. It supports fast, balanced, and accurate integration modes and interactive visualizations.
- `rl`: a configurable, model‑agnostic RL framework that wraps a kinetic model as a Gymnasium environment, so you can define targets, observations, rewards, and actions via a single YAML/JSON config, then train/evaluate agents with SB3.

## Highlights

- **Config‑driven RL**: define actions, observations, rewards, and episode settings in one file.
- **Model‑agnostic**: plug any compatible kinetic model JSON export into the RL environment.
- **Expression‑based features**: safe expressions for observations and rewards.
- **Performance controls**: fast/balanced/accurate simulation modes, step reduction, and optimized ODE settings.
- **Nice plots**: quick interactive Plotly figures from kinetic simulations.

---

## Installation

From your terminal or notebook:

```bash
git clone <this-repo-url>
cd microbesGYM
pip install .  # or: uv pip install .
```

Verify the install and version:

```bash
python -c "import kinetic_model as km; print(km.__version__)"
```

Optional extras (tensorboard, progress bars, visualization, etc.):

```bash
pip install ".[extras]"    # focused extras
pip install ".[all]"       # dev + extras
```

Python >= 3.9 is required.

---

## Repository layout

```text
microbesGYM/
├─ modelTemplates/               # ready-to-use kinetic model JSONs
├─ notebooks/                    # quick demos and tutorials
├─ src/
│  ├─ kinetic_model/             # simulation engine
│  │   ├─ reactor.py             # Reactor/Pulse + ODE integration, plotting
│  │   ├─ bacteria.py, microbiome.py, metabolome.py, ...
│  │   └─ model_from_json.py     # load model from JSON
│  └─ rl/                        # reinforcement learning framework
│      ├─ env.py                 # GeneralMicrobiomeEnv (Gymnasium)
│      ├─ config.py              # `TopLevelConfig` (Pydantic)
│      ├─ model_adapter.py       # bridge kinetic model ⇄ env
│      ├─ wrappers.py            # observation pipeline, normalization, etc.
│      ├─ cli/                   # train/evaluate entry points
│      ├─ docs/                  # RL docs
│      └─ examples/configs/      # ready configs (butyrate, acetate, etc.)
├─ trained_models/               # example checkpoints
├─ VERSION
└─ pyproject.toml
```

---

## Quickstart

### 1) Kinetic simulation in a few lines

```python
from kinetic_model import (
    Metabolite, Metabolome, Microbiome,
    Environment, pH, Temperature, Stirring,
    Pulse, Reactor,
)

# Minimal metabolome and microbiome
glucose = Metabolite("glucose", 10.0, {"C": 6, "H": 12, "O": 6}, "#ff0000")
metabolome = Metabolome([glucose])
microbiome = Microbiome(name="demo", bacteria={})

# Constant environment
env = Environment(pH(metabolome, intercept=7.0), Stirring(rate=0.9), Temperature(37.0))

# One pulse, then simulate
pulse = Pulse(t_start=0.0, t_end=10.0, n_steps=100, environment=env)
reactor = Reactor(microbiome, metabolome, pulses=[pulse], volume=1.0)
reactor.set_balanced_simulation_mode()  # fast|balanced|accurate controls
reactor.simulate()
fig = reactor.make_plot()  # interactive Plotly figure
```

You can also load a full model from JSON:

```python
from kinetic_model import ModelFromJson
model = ModelFromJson("modelTemplates/bh_bt_ri_complete_model_export.json")
# Access: model.metabolome, model.microbiome
```

### 2) Train a controller (RL)

Use a provided config and a model JSON to train with SAC/PPO:

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

Evaluate a trained model (deterministic actions, accurate mode):

```bash
python -m rl.cli.evaluate \
  --algo sac \
  --model-json modelTemplates/bh_bt_ri_complete_model_export.json \
  --config src/rl/examples/configs/butyrate_control.yaml \
  --model-path models/mg_rl_general_sac.zip
```

The evaluation automatically loads observation normalization stats if available
(`*_normalization_stats.json`).

---

## RL configuration model

All RL behavior is captured by `TopLevelConfig` (`src/rl/config.py`) and can be written as YAML/JSON.
Below is a compact YAML example showing the main sections you’ll likely tweak:

```yaml
episode:
  horizon: 250
  dt_hours: 1.0
  training_mode: balanced   # fast|balanced|accurate
simulation:
  min_steps_per_pulse: 10
  steps_per_hour_factor: 50
actions:
  pH_mode: switchable       # controlled|emergent|switchable
  actuators:
    - { name: q,       type: continuous }
    - { name: v,       type: continuous }
    - { name: pH_ctrl, type: binary }
    - { name: pH_set,  type: continuous }
    - { name: stir,    type: continuous }
    - { name: temp,    type: continuous }
  bounds:
    q: [0.0, 0.5]
    v: [0.0, 0.2]
    pH_set: [5.8, 7.8]
    stir: [0.0, 1.0]
    temp: [25.0, 45.0]
observations:
  include: ["met.all", "pH.used", "actuator_echo.all", "met.delta.all", "met.rate.all"]
  pipeline:
    - { normalize: { method: running_mean_var } }
    - { clip: { min: -10, max: 10 } }
rewards:
  error_reward: -1000.0
  terms:
    - { expr: delta_target, weight: 200.0, deadband: 0.02 }
    - { expr: action_q * dt_hours, weight: -0.02 }
    - { expr: action_v, weight: -0.03 }
    - { expr: abs(action_temp - 37), weight: -0.02 }
    - { expr: action_stir, weight: -0.01 }
  terminal:
    - { when: last_step, expr: max(0, delta_target), weight: 15.0 }
target:
  type: metabolite
  name: butyrate
  use_delta: true
```

See `src/rl/docs/DOCUMENTATION.md` for a deeper dive into actions, observations, rewards,
and the environment’s action/observation spaces.

---

## Notebooks and examples

- `notebooks/kinetic_model.ipynb`: build and simulate microbiomes.
- `notebooks/controller_simulation_demo.ipynb`: end-to-end controller demo.
- `src/rl/examples/configs/`: ready-to-run RL configs (`butyrate`, `acetate`, `succinate`, `kombucha`, etc.).
- `modelTemplates/`: curated kinetic model JSONs used by examples.

---

## Command-line reference

- `python -m rl.cli.train_agent --help`
- `python -m rl.cli.evaluate --help`
- `python -m kinetic_model.cli` (prints basic info)

The training script supports SAC, PPO (and optionally TQC if `sb3-contrib` is installed),
TensorBoard logging, normalization sync between train/eval envs, evaluation breakdown logging,
checkpointing with normalization stats, and simple LR/entropy schedules.

---

## Citation

If you use microbesGYM in academic work, please cite this repository. A BibTeX entry will be added once a preprint is available.

---

## Contributing

Issues and PRs are welcome. Please:

- open a descriptive issue first if proposing a larger feature,
- add or update minimal tests if you contribute functionality,
- keep code readable and typed where possible.

---

## License

MIT License © 2025 Daniel Rios Garza
