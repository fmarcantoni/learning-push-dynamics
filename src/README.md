# RBE 577: Physics-Based Learning for Robotic Manipulation
## Project 2: Learning Push Dynamics

**Author:** Filippo Marcantoni (fmarcantoni@wpi.edu), Prahladh Raja (pnamboorkrishnam@wpi.edu)

**Course:** RBE 577 - Machine Learning for Robotics (Prof. Constantinos Chamzas)  

**Institution:** Worcester Polytechnic Institute

---

## Overview

This project implements and compares three modelling paradigms for predicting
the final pose of a rigid object subject to a planar push by a UR10 robotic
manipulator:

1. **Physics-based model** — rigid-body dynamics with numerical integration,
   no training required
2. **Neural network model** — fully connected MLP trained end-to-end on
   push parameter–outcome pairs
3. **Hybrid model** — MLP whose input is augmented with the physics model's
   prediction, combining structured prior knowledge with data-driven learning

All models are trained on an 80/20 train/test split of a cracker box push
dataset. Evaluation reports both Mean Squared Error (MSE) and Mean Euclidean Distance (MED) on
the held-out test set. A gradient-based push planner is also implemented and
evaluated on all three models.

---

## Project Structure
```
project2_fmarcantoni.pdf
project2_fmarcantoni.zip/
├── src/
│   ├── config/
│   │   └── default.yaml      # Hyperparameters, data paths, device config
│   ├── lib/
│   │   ├── physics.py        # Physics engine 
│   │   └── models.py         # Neural networks
│   ├── helpers/
│   │   ├── utils.py          # Data loading utilities 
│   │   └── config.py         # Config handling
│   ├── main.py               # Full training, evaluation and planning pipeline
│   ├── results/                  # Generated plots (auto-created)
│   ├── checkpoints/              # Saved model weights (auto-created)
│   ├── assets/
│   ├── README.md
│   └── requirements.txt
└── data/genesis/
    ├── data_x_cracker_box.npy    # Push parameters [theta0, d, D]
    └── data_y_cracker_box.npy    # Final states [x_f, y_f, theta_f]
```

---

## Installation

### 1. Create Virtual Environment

```bash
conda create -n sfm-env python=3.11 -y
conda activate sfm-env
pip install --upgrade pip
```

or

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install numpy torch matplotlib pyyaml tqdm colorama
```

**Requirements:**
```
torch
numpy
matplotlib
pyyaml
tqdm
colorama
```

---

## Usage

Place your data files in `./data/` then run from the `src/` directory:
```bash
cd src
python main.py
```

To resume from a checkpoint:
```bash
python main.py --checkpoint ./checkpoints/model_epoch_100.pth
```

To use a custom config file:
```bash
python main.py --config ./config/custom.yaml
```

---

## Pipeline

The full pipeline in `main.py` runs the following stages in order:

### 1. Data Loading and Splitting
Raw `.npy` files are loaded and split into train (80%) and test (20%) sets
using a fixed seed via `split_data()`. The train loader is shuffled; the
test loader is unshuffled. The test set is never seen during training.

### 2. Training
Both the NN and Hybrid models are trained simultaneously on the train loader
for the configured number of epochs. The physics model requires no training
and its MSE on each batch is recorded as a fixed reference. Per-epoch MSE
and MED (via `accuracy()`) are logged to the console at the configured
frequency.

### 3. Evaluation on Test Set
`evaluate_planner()` runs all three models over the full test loader in a
single pass, computing MSE via `loss()` and MED via `accuracy()` for each
model. Results are printed in a formatted table.

### 4. Prediction Plots
Two plots are saved to `./results/`:
- `predictions.png` — 3×3 scatter grid (x, y, θ per model) on 200 test samples
- `xy_trajectories.png` — 2D spatial scatter with error lines on 200 test samples

### 5. Push Planning
`plan_push(target)` is called with no model argument, which automatically
runs gradient-based optimization through all three models simultaneously and
returns a dictionary of results. Planning is demonstrated on three held-out
test samples, reporting the optimized push parameters, achieved state, MSE,
and MED for each model.

---

## Implementation Details

### Physics Model (`lib/physics.py`)

Simulates planar rigid-body motion under a sinusoidal velocity profile using
Euler integration over `N=25` timesteps. Key equations:

| Quantity | Formula |
|---|---|
| Moment of inertia | $I = \frac{1}{12}ms^2$ |
| Velocity profile | $v(t) = v_{\max}(\frac{1}{2}\sin(\frac{2\pi t}{T} - \frac{\pi}{2}) + \frac{1}{2})$ |
| Torque | $\tau_i = mv_id$ |
| Angular acceleration | $\alpha_i = \tau_i / I$ |
| Angular update | $\Delta\theta_i = \frac{1}{2}\alpha_i\Delta t^2$ |
| Linear update | $\Delta x_i = -v_i\cos(\theta_i)\Delta t$ |

After integration, local-frame displacements are rotated to the world frame
using the initial push orientation $\theta_0$. The model runs on CPU
(NumPy-compatible) and moves results back to the original device.

> **Note:** 25 simulation steps was found empirically to outperform 100 steps.
> More steps amplify unconstrained angular accumulation (no friction term),
> which is the dominant error source.

### Neural Network Model (`lib/models.py` — `NNModel`)

| Property | Value |
|---|---|
| Architecture | 3 → 32 → 64 → 128 → 128 → 64 → 32 → 3 |
| Activations | ReLU (hidden), none (output) |
| Loss | Mean Squared Error (`loss()`) |
| Metric | Mean Euclidean Distance (`accuracy()`) |
| Optimizer | Adam, lr=0.001 |

The `accuracy()` method computes MED and is used **consistently** across
training monitoring, test-set evaluation, and push planning assessment,
ensuring no metric inconsistencies between pipeline stages.

### Hybrid Model (`lib/models.py` — `NNPhysicsModel`)

Extends `NNModel` with a 6-dimensional input:
```
input = concat([push_params (3D), physics_prediction (3D)])
```

The physics prediction is computed inside `torch.no_grad()` to prevent
spurious gradient flow. Architecture and training procedure are otherwise
identical to the pure NN, isolating the effect of the physics prior.

### Push Planner (`lib/models.py` — `PushPlanner`)

`optimize_push(target, model_name=None)` runs gradient descent through
the frozen forward model to minimise:
```
argmin_x || model(x) - target ||^2
```

When `model_name=None` (default), it automatically runs for all three models
(`"physics"`, `"nn"`, `"hybrid"`) and returns a dict. When called with a
specific name it returns a single tensor, preserving backward compatibility.

---

## Results

### Test Set Evaluation

| Model | Test MSE ↓ | MED (m) ↓ |
|---|---|---|
| Physics | 2.8598 | 2.4359 |
| Neural Network | 0.0075 | 0.0776 |
| **Hybrid** | **0.0036** | **0.0653** |

The hybrid model achieves the best forward-prediction performance on the
held-out test set, with a 2.1× lower MSE and 16% lower MED than the pure NN.

### Push Planning (3 test samples)

| Sample | Model | Plan MSE ↓ | Plan MED (m) ↓ |
|---|---|---|---|
| 0 | Physics | 0.000600 | 0.042430 |
| 0 | NN | **0.000038** | **0.010701** |
| 0 | Hybrid | 0.000439 | 0.036295 |
| 1 | Physics | **0.000001** | **0.001562** |
| 1 | NN | 0.000046 | 0.011781 |
| 1 | Hybrid | 0.000076 | 0.015055 |
| 2 | Physics | **0.000006** | **0.004064** |
| 2 | NN | 0.006630 | 0.141031 |
| 2 | Hybrid | 0.002338 | 0.083741 |

The physics model achieves the best planning accuracy on two of three samples
despite its poor forward-prediction performance, because its smooth loss
landscape enables reliable gradient convergence from zero initialization.
The NN wins on sample 0 but degrades on sample 2, reflecting sensitivity to
the local geometry of the loss landscape.
---

## Limitations and Future Work

- **No friction model** — adding a  viscous damping term to the
  physics engine would eliminate the dominant angular error source
- **Fixed learning rate** — step decay would suppress
  the oscillations visible after epoch 40 in the training curves
- **Residual hybrid** — explicitly predicting
  $\mathbf{y} - \hat{\mathbf{y}}_{\text{physics}}$ would impose a stronger
  inductive bias