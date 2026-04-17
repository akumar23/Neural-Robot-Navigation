# Collision-Detection Neural Net

A PyTorch-based action-conditioned collision-probability predictor for a 2D autonomous-navigation robot. A residual feedforward network scores each discrete steering action for collision risk from raycast-derived features; the controller filters unsafe actions and selects among the remainder using a goal-directed steering behavior, with several recovery subsystems to handle stuck states.

Physics is handled by PyMunk; rendering (optional) by PyGame. Training data is self-collected from a `Wander` exploration policy; the network is trained with focal loss to handle the ~10% collision class imbalance. Monte Carlo Dropout is used at inference to estimate epistemic uncertainty and bias the controller toward conservative actions when the model is unsure.

---

## Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Sensor Model and Feature Engineering](#sensor-model-and-feature-engineering)
4. [Neural Network](#neural-network)
5. [Action Selection and Control Loop](#action-selection-and-control-loop)
6. [Reliability Subsystems](#reliability-subsystems)
7. [Data Collection and Training](#data-collection-and-training)
8. [Simulation Environment](#simulation-environment)
9. [Installation and Usage](#installation-and-usage)
10. [Testing and Evaluation](#testing-and-evaluation)
11. [Repository Layout](#repository-layout)
12. [Performance Characteristics](#performance-characteristics)

---

## Overview

The robot has:

- 5 distance sensors via raycasting at angles `[+66°, +33°, 0°, −33°, −66°]`, range 150 px.
- A discrete action space `A = {−5, −4, …, +4, +5}` (11 steering actions).
- A feature pipeline that maps raw sensors plus robot/goal state to a 20-D vector.
- A residual feedforward network `f_θ: R^20 → R` producing a collision logit per `(state, action)` pair.
- A controller that, each decision cycle, evaluates `f_θ` for all 11 actions, discards those with `σ(f_θ) > τ`, and picks the remaining action closest to the heading demanded by a `Seek` steering behavior.

The network is trained offline on 100k `(features, collision)` pairs collected by a `Wander` policy. The threshold `τ` and several auxiliary subsystems (spatial memory, wall follower, waypoint planner, action smoother) handle cases where the learned policy alone fails (corners, oscillation, dead-ends).

---

## System Architecture

```
 raycast sensors (5)                                goal / robot / velocity
         │                                                   │
         ▼                                                   ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ feature_engineering.engineer_features(...)                    │
 │   → 20-D vector per candidate action a ∈ A                    │
 └───────────────────────────────────────────────────────────────┘
         │
         ▼
 ┌───────────────────────────┐
 │ StandardScaler (20-D)      │   fit once at training time,
 │ models/scaler.pkl          │   pickled for inference
 └───────────────────────────┘
         │
         ▼
 ┌───────────────────────────┐
 │ Action_Conditioned_FF      │   residual FF, 3 res blocks,
 │ input=20, hidden=64        │   LayerNorm, Dropout(0.2)
 │ output: collision logit    │   FocalLoss during training
 └───────────────────────────┘
         │                       (optionally: MC Dropout,
         ▼                        n_samples=10, adjusted
 σ(logit) → p_collision(a)       prediction = μ + λσ)
         │
         ▼
 ┌───────────────────────────┐    ┌──────────────────────────┐
 │ safe-action filter         │◄───│ adaptive threshold τ     │
 │   p(a) < τ                 │    └──────────────────────────┘
 └───────────────────────────┘
         │
         ▼
 ┌───────────────────────────┐    ┌──────────────────────────┐
 │ Seek steering → desired a  │    │ recovery: spatial memory │
 │ pick argmin |desired − a|  │◄───│ wall follower, waypoint  │
 │ among safe actions         │    │ planner, action smoother │
 └───────────────────────────┘    └──────────────────────────┘
         │
         ▼
 PyMunk step × 20 physics substeps per decision
```

One decision cycle = 11 forward passes (one per candidate action) + steering + 20 physics substeps.

---

## Sensor Model and Feature Engineering

### Raw Sensors

Five raycast distances, each clamped to `[0, 150]` pixels:

| Index | Angle   | Name                |
|-------|---------|---------------------|
| 0     | +66°    | `sensor_left_far`   |
| 1     | +33°    | `sensor_left_near`  |
| 2     |   0°    | `sensor_front`      |
| 3     | −33°    | `sensor_right_near` |
| 4     | −66°    | `sensor_right_far`  |

### Feature Vector (20-D Enhanced)

Current production pipeline uses the 20-D vector defined in `src/robot_navigation/feature_engineering.py::engineer_features`. A 12-D legacy mode is retained for backward compatibility and auto-selected when the optional arguments are omitted.

| Idx   | Feature                 | Definition                                                                    | Range       |
|-------|-------------------------|-------------------------------------------------------------------------------|-------------|
| 0–4   | raw sensors             | 5 raycast distances (see above)                                               | `[0, 150]`  |
| 5     | `min_sensor`            | `min(s)`                                                                      | `[0, 150]`  |
| 6     | `sensor_variance`       | `std(s)`                                                                      | `[0, ∞)`    |
| 7     | `front_to_side_ratio`   | `s[2] / ((s[0] + s[4])/2 + ε)`                                                | `[0, ∞)`    |
| 8     | `left_right_asymmetry`  | `mean(s[0:2]) − mean(s[3:5])`                                                 | `(−∞, ∞)`   |
| 9     | `front_clearance`       | `1 / (s[2] + ε)`                                                              | `(0, ∞)`    |
| 10    | `side_gradient`         | `mean(abs(diff(s)))`                                                          | `[0, ∞)`    |
| 11    | `goal_direction`        | `wrap(atan2(dy, dx) − θ_robot) / π`                                           | `[−1, 1]`   |
| 12    | `goal_distance`         | `‖goal − robot‖ / 150`                                                        | `[0, ∞)`    |
| 13    | `velocity_x`            | `v_x / 10`                                                                    | `[−1, 1]`   |
| 14    | `velocity_y`            | `v_y / 10`                                                                    | `[−1, 1]`   |
| 15    | `action_momentum`       | `mean(history[-5:]) / 5`                                                      | `[−1, 1]`   |
| 16    | `action_variance`       | `std(history[-5:]) / 5`                                                       | `[0, 1]`    |
| 17    | `front_goal_alignment`  | `(1 − ‖goal_direction‖) · (s[2] / 150)`                                       | `[0, 1]`    |
| 18    | `escape_urgency`        | `1 / (min(s) + ε)`                                                            | `(0, ∞)`    |
| 19    | `action`                | candidate steering action                                                     | `[−5, +5]`  |

`ε = 1e−6`. `action_history` is a `deque(maxlen=5)`; when length is 0, momentum and variance default to 0. `wrap(·)` normalises to `[−π, π]` via `atan2(sin, cos)`.

### Normalization

A `sklearn.preprocessing.StandardScaler` is fit once at training time on the full 20-D feature matrix (collision label excluded) and persisted to `models/scaler.pkl`. At inference, `scaler.transform(feature_vec.reshape(1, -1))` is applied before the network forward pass. Do not include the collision label when transforming at inference — the scaler was not fit on it.

---

## Neural Network

### Architecture: `Action_Conditioned_FF`

Defined in `src/robot_navigation/networks.py`. Residual feedforward with `LayerNorm` (chosen over `BatchNorm` to support single-sample inference).

```
x ∈ R^20
  ↓
Linear(20 → 64) → LayerNorm → ReLU → Dropout(0.2)
  ↓
[ ResidualBlock(64) ] × 3
  ↓
Linear(64 → 32) → LayerNorm → ReLU → Dropout(0.2)
  ↓
Linear(32 → 1)   ← logit (no sigmoid)
```

Each `ResidualBlock` is:

```
identity = x
x = Linear(64→64) → LayerNorm → ReLU → Dropout(0.2)
x = Linear(64→64) → LayerNorm
x = ReLU(x + identity)
```

The forward method accepts both `(features,)` and `(batch, features)` tensors; single samples are unsqueezed internally and squeezed out before return.

### Output Handling

- The network emits a **logit**. Apply `torch.sigmoid` to obtain `p_collision ∈ [0, 1]`.
- NaN/Inf predictions are clamped to `1.0` (treated as unsafe) by the controller.
- `FocalLoss` internally handles the sigmoid via BCE-with-logits semantics, so no sigmoid is baked into the model.

### Loss: Focal Loss

```python
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)
```

with `α = max(0.1, collision_rate)` (≈ 0.10 on typical collected datasets) and `γ = 2.0`. The implementation computes `p_t` from logits with clamping to `[1e−7, 1 − 1e−7]` for numerical stability. This down-weights easy negatives and concentrates gradient on the ~10% collision minority class.

### Training Configuration

| Hyperparameter   | Value                                                       |
|------------------|-------------------------------------------------------------|
| Optimizer        | Adam                                                        |
| Initial LR       | `1e−2`                                                      |
| LR Scheduler     | `ReduceLROnPlateau(mode=min, factor=0.5, patience=10, min_lr=1e-6)` |
| Batch size       | 32                                                          |
| Epochs           | 100                                                         |
| Loss             | `FocalLoss(α=collision_rate, γ=2.0)`                        |
| Device           | CUDA → MPS → CPU (auto-detected)                            |
| Checkpointing    | Best by test loss → `models/saved_model.pkl`                |

### Monte Carlo Dropout (Inference-Time Uncertainty)

`scripts/run.py::predict_with_uncertainty` toggles the model into `train()` mode (dropout active) under `torch.no_grad()` and runs `n_samples=10` forward passes per action:

```
μ(a) = mean_i σ(f_θ(x_a))
σ̂(a) = std_i  σ(f_θ(x_a))
p_adj(a) = clip(μ(a) + λ · σ̂(a), 0, 1)        λ = uncertainty_penalty_factor
```

`p_adj(a)` replaces the single-pass probability in the safe-action filter. Higher `σ̂` on out-of-distribution states pushes the action over the threshold and makes the controller conservative. Default `λ = 1.0`, `n_samples = 10`.

### LSTM Variant: `Action_Conditioned_LSTM`

`networks.py` also defines a 2-layer LSTM (`hidden=64`, `dropout=0.2`, `batch_first=True`) with the same output head. It accepts `(batch, seq_len, 20)` during training and `(features,)` or `(batch, 1, 20)` at single-timestep inference. A learned hidden state lets the model capture temporal context directly rather than through engineered momentum/variance features. The main `run.py`/`train.py` paths use the feedforward model; the LSTM is available for sequence-level training workflows but is not wired into the production controller.

---

## Action Selection and Control Loop

Per decision cycle (`scripts/run.py::goal_seeking`):

1. Read robot and goal state; compute distance-to-goal progress signal.
2. For each `a ∈ {−5,…,+5}`:
   - Build 20-D features; `scaler.transform`; compute `p_adj(a)` via MC Dropout (or single-pass `σ(logit)` if `use_uncertainty=False`).
   - Accept as safe if `p_adj(a) < τ`.
3. `Seek.get_action(...)` returns the integer action closest to the goal-heading vector.
4. Choose `argmin_{a ∈ safe} |desired − a|`.
5. Apply the chosen action for 20 PyMunk substeps (`simulation_action_repeat = 20`).
6. Update `action_history`, collision counters, stuck counter, and adaptive threshold.

### Adaptive Collision Threshold

Defined in `src/robot_navigation/navigation_config.py` (`NavigationConfig`). Default values:

| Event                                                | τ update                                                                 |
|------------------------------------------------------|--------------------------------------------------------------------------|
| Initial                                              | `τ = 0.3` (`collision_threshold_initial`)                                |
| Progress made (distance − last ≥ 5 px)               | gradually decay toward `0.3` (`collision_threshold_decrease_step`)        |
| No progress for 30 cycles                            | `τ ← min(0.8, τ + 0.15)`                                                 |
| Collision occurred                                   | `τ ← min(0.6, τ + increase_after_collision)`                             |
| `consecutive_no_actions > 3`                         | `τ ← min(0.85, τ + increase_step)` (`collision_threshold_max = 0.85`)    |
| Forced turn-around                                   | `τ = 0.5` (`collision_threshold_after_turn`)                             |
| Goal reached                                         | `τ = 0.3` (reset)                                                        |

The effect is: conservative when progressing, permissive when stuck, reset on goal.

### Stuck Detection

| Trigger                                                        | Action                                       |
|----------------------------------------------------------------|----------------------------------------------|
| `‖Δpos‖ < 3` px over `stuck_counter_max = 8` cycles            | `turn_robot_around()`; reset trackers        |
| No safe actions for `no_safe_actions_turn_threshold = 15`      | `turn_robot_around()`; `τ ← 0.5`             |
| Spatial-memory oscillation (`avg pairwise dist < 30` px, N=15) | force strong turn (`a ∈ {−5,−4,4,5}`)        |
| Action-smoother thrashing (≥3 sign changes in history)         | override with random strong action           |

---

## Reliability Subsystems

These modules augment the learned policy to guarantee liveness in degenerate states.

**`action_smoother.ActionSmoother`** — Maintains an action deque of length 5 and blends `desired_action` with a momentum term weighted `momentum_weight = 0.4`. Detects *thrashing* (rapid sign alternation) and forces a random strong action to break the oscillation.

**`spatial_memory.SpatialMemory`** — Discretises position into `grid_size = 30` px cells with a `decay_rate = 0.97` visit count and a `maxlen = 100` position deque. Exposes repulsion scores per candidate action (proportional to visit count of the predicted next cell) and an `oscillation_detected` flag based on average pairwise distance in the recent position buffer.

**`wall_follower.WallFollower`** — Engaged when `stuck_counter > threshold` or multiple sensors read below the tight-space cutoff. Selects a preferred side (`left`/`right`) from sensor asymmetry, emits turn commands that maintain `target_distance = 70` px from the wall, deactivates after `max_follow_steps = 50` or when open space is detected.

**`waypoint_planner.WaypointPlanner`** — When the direct path to the goal is blocked (front sensors occupied and goal-heading action unsafe), projects a waypoint `waypoint_distance = 120` px along the most-open direction, clamped to the bounded screen area with `boundary_margin = 50`. The waypoint replaces the goal for the `Seek` behavior until reached (`waypoint_reached_threshold = 50` px).

**`openness_scorer.OpennessScorer`** — Per-action scalar score that weights the 5 raw sensor readings by an action-dependent kernel (e.g. strong left turn up-weights the +66° and +33° sensors). Used as a tie-breaker between safe actions with similar goal alignment.

---

## Data Collection and Training

### Collection (`scripts/collect_data.py`)

```python
sim_env = SimulationEnvironment()
wander  = Wander(action_repeat=100)
for i in range(100_000):
    action, force = wander.get_action(i, robot.angle)
    for t in range(100):
        _, collision, sensors = sim_env.step(force)  # sensors read on t == 0
        if collision: break
    features = engineer_features(
        sensors, action,
        robot_pos=robot.position, robot_angle=robot.angle,
        goal_pos=goal.position, velocity=robot.velocity,
        action_history=action_history,
    )
    action_history.append(action)
    row = np.append(features, collision)    # 21 cols
    ...
np.savetxt("data/training_data.csv", rows, delimiter=",")
```

Each decision runs up to 100 physics substeps; on collision, the trajectory for that action is truncated and — if the collision fires in the first 30% of the substep budget — the label is back-propagated to the *previous* action row (the action that caused the terminal state). Output shape `(100000, 21)`. Typical collision rate ≈ 10%.

### Training (`scripts/train.py`)

```
device = cuda | mps | cpu
data   = Data_Loaders(batch_size=32)     # train / test split via data_loaders.py
model  = Action_Conditioned_FF(input_size=20)
α_FL   = max(0.1, collision_rate(data.train_loader))
loss   = FocalLoss(alpha=α_FL, gamma=2.0)
opt    = Adam(model.parameters(), lr=1e-2)
sched  = ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-6)
for epoch in range(100):
    for batch in data.train_loader:
        loss(model(batch["input"]), batch["label"]).backward(); opt.step()
    test_loss = model.evaluate(...)
    sched.step(test_loss)
    if test_loss < best: save(model.state_dict(), "models/saved_model.pkl")
```

The `StandardScaler` is fit inside `data_loaders.Data_Loaders` on the 20-D feature matrix (collision label excluded at column 20) and pickled to `models/scaler.pkl`.

---

## Simulation Environment

Implemented in `src/robot_navigation/simulation.py`.

- **Physics:** PyMunk (Chipmunk2D), Cartesian coordinates, y-up.
- **Rendering:** PyGame, screen coordinates, y-down. Coordinate helpers in `helper.py`:
  - `pm2pgP(point)` / `pg2pmP(point)` — position conversions.
  - `pm2pgV(vec)` — vector (flips y).
- **Robot:** Box-shaped `pm.Body` with `pm.Poly` collision shape, configurable via `NavigationConfig` (mass, speed, friction, turn-rate limits). 5 raycasting sensors attached at the documented angles.
- **Obstacles / goal:** Randomly placed rectangles; goal is a `pm.Body` respawned on reach. Collisions detected via PyMunk shape filters (`categories = 0b1`).
- **Decision cadence:** `action_repeat = 20` physics steps per controller decision (100 during data collection, to produce more diverse trajectory segments per action).
- **Headless mode:** `simulation.py` sets `HEADLESS = True` and `os.environ['SDL_VIDEODRIVER'] = 'dummy'` at import time. Required for unattended training / test runs.

---

## Installation and Usage

### Requirements

Python ≥ 3.8. From `requirements.txt`:

```
pymunk
numpy
noise
torch
matplotlib
scikit-learn
pygame
```

### Install

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Collect Data

```bash
python scripts/collect_data.py
# → data/training_data.csv  (100_000 × 21)
```

### Train

```bash
python scripts/train.py
# → models/saved_model.pkl, models/scaler.pkl
```

`train.py` auto-selects CUDA, Apple MPS, or CPU. Expect ~100 epochs; best-by-test-loss checkpointing.

### Run (Trained Model)

```bash
python scripts/run.py
```

By default runs `goal_seeking(goals_to_reach=2, use_uncertainty=True, uncertainty_penalty_factor=1.0, n_mc_samples=10)`. Toggle `use_uncertainty=False` for single-pass inference.

### Test

```bash
python tests/test_robot_movement.py                              # default: 5 tests × 1000 iters
python tests/test_robot_movement.py --tests 10 --iterations 2000
python tests/test_robot_movement.py --quiet
```

### Unit Tests

```bash
python tests/test_openness_scorer.py
python tests/test_spatial_memory.py
python tests/test_wall_follower_unit.py
python tests/test_waypoint_planner_unit.py
```

---

## Testing and Evaluation

`tests/test_robot_movement.py` is the primary end-to-end harness. It instantiates the simulator, loads the trained model, and runs `N` independent trials (random obstacle/goal placement). Per-trial metrics:

| Metric                    | Definition                                                    |
|---------------------------|---------------------------------------------------------------|
| Success rate              | fraction of trials reaching the goal before iteration timeout |
| Avg iterations to goal    | mean over successful trials                                   |
| Collision count           | total collision events per trial                              |
| Avg progress              | final distance-reduction / initial distance                   |
| Stuck events              | count of triggered turn-around / wall-follower activations    |

Harness integrates the same recovery subsystems as `run.py`, including action-history propagation into `engineer_features` and MC Dropout uncertainty estimation.

---

## Repository Layout

```
Collision-Detection-Neural-Net/
├── src/robot_navigation/
│   ├── simulation.py           # PyMunk world, robot, raycasting, collision detection
│   ├── networks.py             # Action_Conditioned_FF, Action_Conditioned_LSTM, FocalLoss
│   ├── feature_engineering.py  # 12-D legacy / 20-D enhanced feature pipeline
│   ├── steering.py             # Seek (goal) and Wander (exploration) behaviors
│   ├── action_smoother.py      # Momentum smoothing + thrashing detection
│   ├── spatial_memory.py       # Grid visit counts, oscillation detection
│   ├── wall_follower.py        # Wall-following escape controller
│   ├── openness_scorer.py      # Per-action open-space scoring
│   ├── waypoint_planner.py     # Intermediate-goal generation
│   ├── navigation_config.py    # NavigationConfig dataclass (all tuning constants)
│   ├── data_loaders.py         # Data_Loaders: splits, scaler fit, DataLoader wrapping
│   └── helper.py               # Vector/angle utilities, coordinate conversions
├── scripts/
│   ├── collect_data.py         # Wander-based 100k sample collection
│   ├── train.py                # Training loop, FocalLoss, ReduceLROnPlateau
│   └── run.py                  # Inference loop with MC Dropout + recovery stack
├── tests/
│   ├── test_robot_movement.py  # End-to-end navigation harness
│   ├── test_openness_scorer.py
│   ├── test_spatial_memory.py
│   ├── test_wall_follower_unit.py
│   └── test_waypoint_planner_unit.py
├── models/                     # saved_model.pkl, scaler.pkl
├── data/                       # training_data.csv
├── assets/                     # robot.png, robot_inverse.png, demo.gif
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Performance Characteristics

Measured on the default simulation (1080 × 900 px, random obstacle layout, sensor range 150 px, action repeat 20):

| Metric                        | Value                      |
|-------------------------------|----------------------------|
| Goal-reach success rate       | 75–80%                     |
| Avg iterations per goal       | 800–1300                   |
| Collision rate in training data | ~10%                     |
| Sensor range                  | 150 px                     |
| Action space                  | 11 discrete values, `[−5, +5]` |
| Action repeat                 | 20 physics steps / decision|
| Training time (100 epochs)    | ~5–10 min on CPU / MPS     |
| Inference per decision        | 11 forward passes (× 10 MC samples if uncertainty enabled) |

The controller is deterministic given fixed RNG seeds in both PyMunk and NumPy; MC Dropout is the sole source of inference-time stochasticity.

---

![demo](assets/demo.gif)
