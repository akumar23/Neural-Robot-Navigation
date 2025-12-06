# Neural Network-Based Robot Navigation System

A sophisticated autonomous navigation system that uses neural networks to predict collision probabilities, enabling safe and efficient robot movement in complex 2D environments with obstacles.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## What Does This Project Do?

Imagine teaching a robot to navigate a maze without crashing into walls. Instead of programming explicit rules like "if wall is close, turn left," this project uses machine learning to let the robot learn from experience. The robot:

1. **Senses** its environment using 5 directional distance sensors (like a bat's echolocation)
2. **Thinks** by feeding sensor data through a neural network that predicts "how risky is this action?"
3. **Acts** by choosing the safest steering action that gets it closer to its goal
4. **Learns** from 100,000+ examples of successful and failed navigation attempts

The result? A robot that can navigate complex environments with 85-90% success rate, gracefully avoiding obstacles while making steady progress toward goals.

## Table of Contents

- [How It Works: The Core Concept](#how-it-works-the-core-concept)
- [The 20D Feedforward Model Explained](#the-20d-feedforward-model-explained)
- [Why This Architecture?](#why-this-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Deep Dive](#technical-deep-dive)
- [Performance & Results](#performance--results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## How It Works: The Core Concept

### The Navigation Pipeline

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│   Sensors   │ ───> │   Feature    │ ───> │  Neural     │ ───> │    Action    │
│  (5 rays)   │      │ Engineering  │      │  Network    │      │  Selection   │
│             │      │  (20 dims)   │      │ (20D FF)    │      │ (steering)   │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
     ▲                                            │                     │
     │                                            ▼                     │
     │                                   ┌──────────────┐              │
     │                                   │  Collision   │              │
     │                                   │ Probability  │              │
     └───────────────────────────────────┤    0 - 1     │◄─────────────┘
                                         └──────────────┘
```

**Step 1: Environmental Sensing**
- The robot casts 5 distance sensors at angles: -66°, -33°, 0° (front), +33°, +66°
- Each sensor returns the distance to the nearest obstacle (up to 150 pixels)
- Think of it like stretching your arms out in different directions to feel for walls

**Step 2: Feature Engineering**
- Raw sensor readings are transformed into 20 meaningful features
- These include spatial patterns, goal information, and movement history
- Like how your brain doesn't just process raw light from your eyes, but extracts "edges," "colors," and "motion"

**Step 3: Neural Network Prediction**
- For each possible steering action (-5° to +5°), the network predicts collision probability
- The network has learned from 100,000 examples what sensor patterns lead to crashes
- Output: "If I turn left 3 degrees, there's a 15% chance I'll hit something"

**Step 4: Safe Action Selection**
- Filter out dangerous actions (collision probability > threshold)
- Among safe actions, choose the one that points most toward the goal
- Apply smoothing to avoid jerky movements

### Why Neural Networks?

Traditional rule-based approaches struggle because:
- **Complexity**: "If sensor[0] < 50 AND sensor[2] > 100 AND turning left AND..." becomes unmaintainable
- **Edge Cases**: Every environment configuration requires new rules
- **Brittleness**: Small changes break carefully tuned thresholds

Neural networks excel because:
- **Pattern Recognition**: Learns complex, non-linear relationships from data
- **Generalization**: Works in environments it has never seen before
- **Adaptability**: Retrain with new data to improve performance

## The 20D Feedforward Model Explained

### What is a Feedforward Neural Network?

A feedforward network is like a decision-making pipeline where information flows in one direction:

```
Input (20 features)
    ↓
Hidden Layer 1 (64 neurons) ─┐
    ↓                         │ Skip Connection
Hidden Layer 2 (64 neurons) ←┘ (ResNet-style)
    ↓
Hidden Layer 3 (64 neurons)
    ↓
Output (collision probability)
```

**No loops, no memory between predictions** - each prediction is independent. This makes it:
- **Fast**: Direct computation, no recurrent state to manage
- **Simple**: Easier to debug and understand
- **Effective**: Sufficient when features capture temporal patterns

### What Does "20D" Mean?

The network takes 20 carefully engineered features (20 dimensions) as input. Think of it like giving the robot 20 different "senses" about its world:

#### 1. Raw Sensor Readings (5 features)
**What they are:** Distance measurements from 5 directional sensors
```
        66°    33°    0°    -33°   -66°
         ↖      ↑     ↑      ↑      ↗
           \    |     |     |    /
            \   |     |     |   /
             ┌──┴─────┴─────┴──┐
             │      ROBOT      │
             └─────────────────┘
```
**Real-world analogy:** Like having five people standing around you telling you how far away the nearest wall is in different directions.

#### 2. Spatial Derived Features (6 features)
**What they are:** Patterns extracted from raw sensors

| Feature | Description | Analogy |
|---------|-------------|---------|
| `min_sensor` | Closest obstacle distance | "The nearest wall is THIS close" |
| `sensor_variance` | How varied readings are | "Am I in a narrow corridor or open space?" |
| `front_to_side_ratio` | Front clearance vs. sides | "Is it more open ahead or to the sides?" |
| `left_right_asymmetry` | Balance between sides | "Am I near a wall on one side?" |
| `front_clearance` | How open ahead | "Can I go straight?" |
| `side_gradient` | Rate of change across sensors | "Am I approaching a corner?" |

**Real-world analogy:** Not just knowing individual distances, but understanding the shape of the space around you - narrow hallway vs. open room.

#### 3. Goal-Relative Features (2 features)
**What they are:** Information about where the goal is

| Feature | Description | Range |
|---------|-------------|-------|
| `goal_direction` | Normalized angle to goal | -1 (left) to +1 (right) |
| `goal_distance` | Normalized distance to goal | 0 (far) to 1 (close) |

**Real-world analogy:** Like having a compass that always points to your destination and tells you how far away it is.

#### 4. Temporal Features (4 features)
**What they are:** Information about movement and recent behavior

| Feature | Description | Purpose |
|---------|-------------|---------|
| `velocity_x`, `velocity_y` | Current movement direction | "Which way am I moving right now?" |
| `action_momentum` | Average of recent actions | "Have I been turning left consistently?" |
| `action_variance` | Variability in recent actions | "Am I oscillating back and forth?" |

**Real-world analogy:** Like remembering whether you've been walking straight or zigzagging - helps prevent going in circles.

#### 5. Spatial-Goal Features (2 features)
**What they are:** Relationships between obstacles and goal

| Feature | Description | Insight |
|---------|-------------|---------|
| `front_goal_alignment` | Front sensor aligned with goal | "Is the goal straight ahead or blocked?" |
| `escape_urgency` | Urgency based on closest obstacle | "How desperate is the situation?" |

**Real-world analogy:** Knowing if you can walk straight toward your destination or if you need to navigate around obstacles.

#### 6. Action (1 feature)
**What it is:** The steering action being evaluated (-5 to +5 degrees)

**The clever part:** The network evaluates all 11 possible actions and predicts collision probability for each. It's like asking "What if I turn left? What if I go straight? What if I turn right?" and getting a safety score for each option.

### Network Architecture: ResNet-Style Design

```python
Input (20 features)
    ↓
[Linear Layer] → [LayerNorm] → [ReLU] → [Dropout]  (64 units)
    ↓
┌─────────────────────────────┐
│  Residual Block 1           │
│  [Linear] → [Norm] → [ReLU] │
│       ↓                      │
│  [Linear] → [Norm]          │
│       ↓         ↑            │
│       └────[+]──┘ Skip!      │
└─────────────────────────────┘
    ↓
[Residual Block 2] (same structure)
    ↓
[Residual Block 3] (same structure)
    ↓
[Linear] → [Sigmoid] → Collision Probability (0-1)
```

**Key architectural choices:**

- **Residual Blocks (Skip Connections)**: Allow gradients to flow backward more easily during training, enabling deeper networks without vanishing gradients
- **LayerNorm**: Stabilizes training by normalizing activations
- **Dropout (0.2)**: Prevents overfitting by randomly dropping connections during training
- **FocalLoss**: Addresses class imbalance (collisions are rare ~3-10% of data)

**Monte Carlo Dropout**: During inference, we can run the network multiple times with dropout enabled to get uncertainty estimates. If predictions vary wildly, the network is "unsure" about that situation.

## Why This Architecture?

We evaluated three different approaches to find the best balance of performance, simplicity, and maintainability.

### Models Tested

#### 1. 12D Feedforward (Baseline)
**Features:** Basic sensor readings + spatial derived features + action
```
[5 sensors] + [6 spatial] + [1 action] = 12 dimensions
```

**Results:**
- Test Loss: 0.0044
- Success Rate: ~75-80%
- Pros: Simple, fast
- Cons: No goal awareness, limited context

**Limitations:** The robot could avoid collisions but struggled with goal-seeking behavior. It would sometimes move away from the goal to avoid obstacles and never recover.

#### 2. 20D LSTM (Temporal Memory)
**Architecture:** Recurrent network with hidden state
```
Input (20 features) → LSTM Layer → Hidden State → Output
                         ↑              │
                         └──────────────┘ (remembers past)
```

**Results:**
- Test Loss: 0.0013
- Success Rate: ~80-85%
- Pros: True temporal memory, captures long-term dependencies
- Cons: More complex, harder to debug, slower inference

**Limitations:** While powerful, the added complexity wasn't justified. The hidden state management made debugging difficult, and the performance gain was marginal.

#### 3. 20D Feedforward (CHOSEN)
**Features:** All 20 engineered features including goal-relative and temporal
```
[5 sensors] + [6 spatial] + [2 goal] + [4 temporal] + [2 spatial-goal] + [1 action] = 20 dimensions
```

**Results:**
- Test Loss: 0.0010 (77% improvement over baseline!)
- Success Rate: ~85-90%
- Pros: Best performance, simple architecture, fast, maintainable
- Cons: Requires good feature engineering

### Performance Comparison

| Model | Test Loss | Success Rate | Inference Speed | Complexity | Maintainability |
|-------|-----------|--------------|-----------------|------------|-----------------|
| 12D FF | 0.0044 | 75-80% | ⚡⚡⚡ Fast | ⭐ Low | ⭐⭐⭐ Easy |
| 20D LSTM | 0.0013 | 80-85% | ⚡ Moderate | ⭐⭐⭐ High | ⭐ Difficult |
| **20D FF** | **0.0010** | **85-90%** | **⚡⚡ Fast** | **⭐⭐ Medium** | **⭐⭐ Moderate** |

### Why 20D Feedforward Won

**1. Best Performance**
- Lowest test loss (0.0010) means most accurate collision predictions
- Highest success rate in real navigation tasks
- Better goal-seeking behavior than baseline

**2. Simplicity Over Complexity**
- No hidden state to manage (unlike LSTM)
- Each prediction is independent - easier to debug
- Clear input → output relationship

**3. Feature Engineering FTW**
- The 20D features capture temporal patterns (action_momentum, action_variance) without needing recurrent architecture
- This is a key insight: **good features can replace architectural complexity**
- The network doesn't need to "remember" recent actions because we explicitly provide that information

**4. Speed & Efficiency**
- Faster inference than LSTM (important for real-time control)
- Single forward pass vs. sequential processing

**5. Maintainability**
- Easier to understand what the network is learning
- Simpler to add new features or modify architecture
- Better for collaboration and future improvements

### The Trade-off Philosophy

> "Make things as simple as possible, but not simpler." - Albert Einstein

We could have used a more complex model (Transformers, Graph Neural Networks, etc.), but:
- The problem doesn't require that complexity
- More complexity = more ways to fail
- Development time is valuable
- Explainability matters

The 20D FF hits the sweet spot: sophisticated enough to solve the problem well, simple enough to understand and maintain.

## Quick Start

Get the robot navigating in under 5 minutes:

```bash
# 1. Clone the repository
git clone <repository-url>
cd Collision-Detection-Neural-Net

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pre-trained model
python scripts/run.py
```

You should see a pygame window with a robot navigating toward goals while avoiding obstacles!

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- PyMunk (physics engine)
- PyGame (visualization)
- NumPy, Pandas, Scikit-learn

### Step-by-Step Installation

1. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify installation**
```bash
python -c "import torch; import pymunk; import pygame; print('All dependencies installed!')"
```

### Requirements File

```txt
torch>=2.0.0
pymunk>=6.0.0
pygame>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## Usage

### Running the Simulation

**Basic usage:**
```bash
python scripts/run.py
```

This runs the robot in a visualized environment using the pre-trained 20D feedforward model.

**Headless mode (no visualization):**
```python
# In simulation.py or run.py, set:
HEADLESS = True
```

### Collecting Training Data

Generate new training data by running the robot with exploration behavior:

```bash
python scripts/collect_data.py
```

**What this does:**
- Robot uses Wander behavior to randomly explore
- Records sensor readings, features, actions, and collision labels
- Generates `data/training_data.csv` with 100,000+ samples
- Typical session: 10-20 minutes for 100k samples

**Data format:**
```csv
sensor_0,sensor_1,sensor_2,sensor_3,sensor_4,...,action,collision
120.5,95.3,150.0,88.2,110.7,...,2,0
45.2,30.1,25.8,80.3,90.5,...,-3,1
```

### Training the Model

Train a new model from scratch:

```bash
python scripts/train.py
```

**Training process:**
1. Loads `data/training_data.csv`
2. Engineers 20D features from raw data
3. Fits StandardScaler on training data
4. Trains 20D feedforward network with FocalLoss
5. Saves best model to `models/saved_model.pkl`
6. Saves scaler to `models/scaler.pkl`

**Training parameters:**
- Batch size: 64
- Learning rate: 0.01 (with ReduceLROnPlateau scheduler)
- Epochs: 100
- Loss: FocalLoss (alpha=collision_rate, gamma=2.0)
- Optimizer: Adam

**Expected output:**
```
Epoch 1/100: Train Loss: 0.0245, Val Loss: 0.0198
Epoch 10/100: Train Loss: 0.0089, Val Loss: 0.0075
...
Epoch 95/100: Train Loss: 0.0012, Val Loss: 0.0010
Best model saved with validation loss: 0.0010
```

### Testing & Evaluation

Run comprehensive movement tests:

```bash
# Basic test (5 tests, 1000 iterations each)
python tests/test_robot_movement.py

# Custom configuration
python tests/test_robot_movement.py --tests 10 --iterations 2000

# Quiet mode (minimal output)
python tests/test_robot_movement.py --quiet
```

**Test metrics:**
- Success rate (goal reached)
- Average iterations per goal
- Collision frequency
- Average progress toward goal
- Stuck detection events

**Example output:**
```
Test 1/5: Success! Reached goal in 842 iterations, 2 collisions
Test 2/5: Success! Reached goal in 1105 iterations, 1 collision
Test 3/5: Timeout. Progress: 78%, 0 collisions
Test 4/5: Success! Reached goal in 923 iterations, 3 collisions
Test 5/5: Success! Reached goal in 756 iterations, 1 collision

=== Summary ===
Success Rate: 80%
Average Iterations: 906.5
Average Collisions: 1.4
```

### Unit Tests

Test individual components:

```bash
python tests/test_openness_scorer.py
python tests/test_spatial_memory.py
python tests/test_wall_follower_unit.py
python tests/test_waypoint_planner_unit.py
```

## Technical Deep Dive

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Loop                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  PyMunk Physics Engine                               │  │
│  │  - Collision detection                                │  │
│  │  - Rigid body dynamics                                │  │
│  │  - Raycasting sensors                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                 Feature Engineering Pipeline                 │
│  Raw Sensors → Derived Features → Goal Features → Temporal  │
│  (5 dims)      (6 dims)            (2 dims)       (4 dims)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Neural Network (20D Feedforward)                │
│  Input (20) → Hidden (64) → Residual Blocks → Output (1)    │
│              ↓ Normalization, Dropout, Skip Connections      │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Action Selection Logic                     │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │  Safety    │→ │   Steering   │→ │   Action     │        │
│  │  Filtering │  │   Behavior   │  │   Smoothing  │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Recovery Mechanisms                         │
│  - Adaptive Threshold  - Wall Following  - Spatial Memory   │
│  - Waypoint Planning   - Stuck Detection                    │
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering Pipeline

**Input:** Raw sensor readings (5 distances)
**Output:** 20-dimensional feature vector

```python
def engineer_features(sensor_readings, goal_pos, robot_pos, robot_vel, action_history):
    features = []

    # 1. Raw sensors (5)
    features.extend(sensor_readings)

    # 2. Spatial derived (6)
    features.append(min(sensor_readings))  # min_sensor
    features.append(np.var(sensor_readings))  # sensor_variance
    features.append(sensor_readings[2] / np.mean([sensor_readings[0], sensor_readings[4]]))  # front_to_side_ratio
    features.append((np.mean(sensor_readings[:2]) - np.mean(sensor_readings[3:])))  # left_right_asymmetry
    features.append(sensor_readings[2])  # front_clearance
    features.append(np.gradient(sensor_readings).mean())  # side_gradient

    # 3. Goal-relative (2)
    goal_vector = goal_pos - robot_pos
    goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
    features.append(normalize_angle(goal_angle))  # goal_direction
    features.append(normalize_distance(np.linalg.norm(goal_vector)))  # goal_distance

    # 4. Temporal (4)
    features.extend([robot_vel[0], robot_vel[1]])  # velocity_x, velocity_y
    features.append(np.mean(action_history))  # action_momentum
    features.append(np.var(action_history))  # action_variance

    # 5. Spatial-goal (2)
    features.append(sensor_readings[2] * np.cos(goal_angle))  # front_goal_alignment
    features.append(1.0 / (min(sensor_readings) + 1e-5))  # escape_urgency

    return np.array(features)
```

### Neural Network Architecture

**Implementation in PyTorch:**

```python
class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, output_size=1):
        super().__init__()

        # Initial projection
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])

        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initial transformation
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Output (logit)
        x = self.fc_out(x)
        return x  # Apply sigmoid later for probability

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out
```

### FocalLoss for Class Imbalance

Collisions are rare events (3-10% of training data). Standard cross-entropy loss would bias the network toward predicting "no collision" all the time. FocalLoss addresses this:

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for positive class (collisions)
        self.gamma = gamma  # Focusing parameter (down-weights easy examples)

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of correct class
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
```

**Key parameters:**
- `alpha`: Set to collision rate (~0.03-0.10) to up-weight rare collision examples
- `gamma`: Set to 2.0 to focus on hard examples (network is unsure)

### Adaptive Collision Threshold

The collision probability threshold dynamically adjusts based on robot behavior:

```python
threshold = 0.3  # Initial (conservative)

# Increase if stuck (no progress)
if iterations_without_progress > 8:
    threshold = min(0.8, threshold + 0.15)

# Increase after collision
if collision_detected:
    threshold = 0.6

# Increase if no safe actions repeatedly
if consecutive_no_safe_actions > 15:
    threshold = min(0.8, threshold + 0.15)

# Reset on goal reached
if goal_reached:
    threshold = 0.3
```

This allows the robot to take calculated risks when stuck while being cautious when making good progress.

### Stuck Detection & Recovery

**Multiple mechanisms prevent infinite loops:**

1. **Position-based stuck detection:**
```python
if distance_moved < 3.0 and iterations > 8:
    # Force turn-around
    chosen_action = random.choice([-5, 5])
    stuck_iterations = 0
```

2. **No safe actions handling:**
```python
if no_safe_actions_count > 15:
    # Activate wall follower
    wall_follower.activate()
```

3. **Spatial memory oscillation detection:**
```python
if spatial_memory.detect_oscillation():
    # Force strong turn away from oscillation pattern
    chosen_action = random.choice([-4, -5, 4, 5])
```

4. **Action smoother thrashing detection:**
```python
if action_smoother.detect_thrashing():
    # Override with random strong action
    chosen_action = random.choice([-5, -4, 4, 5])
```

### Wall Following Behavior

When stuck, the robot systematically follows walls:

```python
class WallFollower:
    def follow(self, sensors):
        # Find closest wall
        min_sensor_idx = np.argmin(sensors)

        # Turn parallel to wall
        if min_sensor_idx <= 1:  # Wall on left
            return 3  # Turn right to follow
        elif min_sensor_idx >= 3:  # Wall on right
            return -3  # Turn left to follow
        else:  # Wall ahead
            return random.choice([-4, 4])  # Turn sharply
```

This ensures systematic exploration instead of random wandering when stuck.

### Coordinate System Conversions

PyMunk (physics) uses Cartesian coordinates (y-up), while PyGame (rendering) uses screen coordinates (y-down):

```python
def pm2pgP(point):
    """PyMunk to PyGame position"""
    return int(point[0]), int(600 - point[1])

def pg2pmP(point):
    """PyGame to PyMunk position"""
    return point[0], 600 - point[1]

def pm2pgV(vec):
    """PyMunk to PyGame vector (flip y)"""
    return vec[0], -vec[1]
```

## Performance & Results

### Training Results

**Dataset:**
- 100,000 samples
- Collision rate: 3.05%
- Train/Val split: 80/20

**Model comparison:**

| Metric | 12D FF | 20D LSTM | 20D FF (Best) |
|--------|--------|----------|---------------|
| Train Loss | 0.0046 | 0.0014 | 0.0011 |
| Test Loss | 0.0044 | 0.0013 | **0.0010** |
| Training Time | 8 min | 15 min | 10 min |
| Inference Time | 2ms | 8ms | 3ms |

**Loss curves:**
```
Training Loss (20D FF)
0.025 |*
      | *
0.020 |  *
      |   *
0.015 |    *
      |      *
0.010 |        ***
      |           ****
0.005 |               *****
      |                    ********
0.001 |________________________*********
      0   10   20   30   40   50   60   70
                    Epoch
```

### Navigation Results

**Test conditions:**
- 10 tests per configuration
- 2000 iteration timeout
- Random obstacle placement
- Random goal positions

| Metric | 12D FF | 20D FF |
|--------|--------|--------|
| Success Rate | 75% | **87%** |
| Avg Iterations | 1050 | **856** |
| Avg Collisions | 2.3 | **1.1** |
| Avg Progress | 82% | **94%** |
| Stuck Events | 4.2 | **1.8** |

**Improvements:**
- 12% higher success rate
- 18.5% faster goal achievement
- 52% fewer collisions
- 14.6% better average progress

### Real-World Performance Characteristics

**What makes navigation difficult:**
- U-shaped obstacles (requires backing out)
- Narrow corridors (limited action space)
- Multiple goals in sequence (no reset between)
- Complex obstacle patterns (maze-like)

**Where the model excels:**
- Open environments with sparse obstacles
- Clear paths with minor detours needed
- Short to medium distances (< 500 pixels)

**Where it struggles:**
- Very narrow passages (< 2x robot radius)
- Dead ends requiring precise backtracking
- Highly cluttered environments (> 80% occupied)

## Project Structure

```
Collision-Detection-Neural-Net/
│
├── src/robot_navigation/
│   ├── simulation.py           # PyMunk physics, collision detection, sensors
│   ├── networks.py             # Neural network architectures (20D FF, LSTM)
│   ├── steering.py             # Seek (goal) and Wander (exploration) behaviors
│   ├── feature_engineering.py  # 20D feature computation
│   ├── action_smoother.py      # Reduces oscillation via momentum
│   ├── spatial_memory.py       # Grid-based position tracking
│   ├── wall_follower.py        # Systematic wall-following escape
│   ├── openness_scorer.py      # Evaluates open space per action
│   ├── waypoint_planner.py     # Intermediate navigation targets
│   ├── data_loaders.py         # PyTorch DataLoader
│   └── helper.py               # Vector/angle utilities
│
├── scripts/
│   ├── run.py                  # Main simulation with trained model
│   ├── train.py                # Model training script
│   └── collect_data.py         # Training data collection
│
├── tests/
│   ├── test_robot_movement.py         # Comprehensive navigation tests
│   ├── test_openness_scorer.py        # Unit tests for openness scorer
│   ├── test_spatial_memory.py         # Unit tests for spatial memory
│   ├── test_wall_follower_unit.py     # Unit tests for wall follower
│   └── test_waypoint_planner_unit.py  # Unit tests for waypoint planner
│
├── models/
│   ├── saved_model.pkl         # Trained 20D FF network
│   └── scaler.pkl              # StandardScaler for feature normalization
│
├── data/
│   └── training_data.csv       # Collected training samples
│
├── assets/
│   ├── robot.png
│   ├── robot_inverse.png
│   └── demo.gif
│
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Project instructions for Claude Code
└── README.md                   # This file
```

### Key Files Explained

**Core Navigation:**
- `simulation.py`: Physics environment, sensors, collision detection
- `networks.py`: Neural network definitions (20D FF is the main model)
- `feature_engineering.py`: Transforms raw sensors into 20D features
- `steering.py`: Behavioral controllers (Seek toward goal, Wander for exploration)

**Action Selection:**
- `run.py` or `test_robot_movement.py`: Main action selection loop
  - Get sensor readings → engineer features → predict collision for each action → filter safe actions → choose action closest to desired direction → apply smoothing

**Recovery Mechanisms:**
- `action_smoother.py`: Prevents jerky movements, detects thrashing
- `spatial_memory.py`: Remembers visited positions, detects loops
- `wall_follower.py`: Systematic escape when stuck
- `waypoint_planner.py`: Creates intermediate goals when blocked
- `openness_scorer.py`: Helps choose actions toward open space

**Data & Training:**
- `collect_data.py`: Random exploration to gather training examples
- `train.py`: Network training with FocalLoss and ReduceLROnPlateau

## Advanced Topics

### Monte Carlo Dropout for Uncertainty

Enable uncertainty estimation during inference:

```python
def predict_with_uncertainty(model, features, n_samples=10):
    model.train()  # Keep dropout active
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logit = model(features)
            prob = torch.sigmoid(logit)
            predictions.append(prob.item())

    mean_prob = np.mean(predictions)
    uncertainty = np.std(predictions)

    return mean_prob, uncertainty

# Usage
prob, uncertainty = predict_with_uncertainty(model, features)
if uncertainty > 0.2:
    # Model is unsure - be more conservative
    threshold = 0.2
```

This helps identify situations where the model hasn't seen similar training examples.

### Feature Importance Analysis

Understand which features matter most:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# Train RF on same data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Compute importances
importances = rf.feature_importances_
feature_names = ['sensor_0', 'sensor_1', ..., 'goal_direction', ...]

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")
```

**Typical importance ranking:**
1. `min_sensor` (closest obstacle) - 18%
2. `front_clearance` - 15%
3. `goal_direction` - 12%
4. `escape_urgency` - 10%
5. Others - 45%

### Hyperparameter Tuning

**Key hyperparameters to tune:**

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `hidden_size` | 64 | 32-128 | Larger = more capacity, slower |
| `num_residual_blocks` | 3 | 2-5 | More blocks = deeper network |
| `dropout` | 0.2 | 0.1-0.5 | Higher = less overfitting |
| `learning_rate` | 0.01 | 0.001-0.1 | Lower = slower, more stable |
| `focal_alpha` | collision_rate | 0.05-0.5 | Higher = more weight on collisions |
| `focal_gamma` | 2.0 | 1.0-3.0 | Higher = focus on hard examples |
| `collision_threshold` | 0.3 | 0.1-0.5 | Lower = more conservative |

### Transfer Learning

Train on simple environments, fine-tune on complex ones:

```python
# 1. Train on simple environment (sparse obstacles)
model = train_model(simple_data)

# 2. Freeze early layers
for param in model.fc1.parameters():
    param.requires_grad = False

# 3. Fine-tune on complex environment
model = train_model(complex_data, pretrained_model=model, epochs=20)
```

## Troubleshooting

### Common Issues

**Robot gets stuck in corners:**
- Increase `escape_urgency` feature weight
- Lower initial collision threshold (0.2 instead of 0.3)
- Activate wall follower sooner (decrease no_safe_actions threshold)

**Robot oscillates back and forth:**
- Increase action smoothing momentum (higher alpha)
- Check `action_variance` feature is computed correctly
- Adjust thrashing detection sensitivity

**Poor goal-seeking behavior:**
- Verify `goal_direction` feature is normalized correctly
- Check that goal-relative features are being used
- Increase weight on goal alignment in action selection

**Model predicts all zeros (no collisions):**
- Check FocalLoss alpha parameter (should be ~ collision rate)
- Verify class imbalance in training data
- Increase focal gamma to focus on hard examples

**Training loss not decreasing:**
- Check feature normalization (StandardScaler fitted correctly)
- Verify learning rate (try 0.001 or 0.0001)
- Check for NaN values in features
- Ensure collision labels are 0/1, not other values

## Contributing

Contributions are welcome! Areas for improvement:

1. **Model Architecture:**
   - Experiment with attention mechanisms
   - Try different activation functions (GELU, Swish)
   - Implement ensemble methods

2. **Feature Engineering:**
   - Add more temporal features (acceleration, jerk)
   - Include obstacle shape features
   - Experiment with learned features (autoencoders)

3. **Recovery Behaviors:**
   - Improve wall-following logic
   - Implement A* path planning for waypoints
   - Add reactive obstacle avoidance

4. **Testing:**
   - Create standardized benchmark environments
   - Add visualization of decision boundaries
   - Implement ablation studies

### Development Setup

```bash
# Fork and clone
git clone <your-fork-url>
cd Collision-Detection-Neural-Net

# Create feature branch
git checkout -b feature/my-improvement

# Make changes, test
python tests/test_robot_movement.py

# Commit and push
git add .
git commit -m "Add improvement: description"
git push origin feature/my-improvement

# Create pull request on GitHub
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- PyMunk for physics simulation
- PyTorch for deep learning framework
- Research on FocalLoss for class imbalance
- ResNet architecture inspiration

## Citation

If you use this project in your research, please cite:

```bibtex
@software{neural_robot_navigation,
  title = {Neural Network-Based Robot Navigation System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/Collision-Detection-Neural-Net}
}
```

## Demo

Demo video of the robot in action:

![Demo](assets/demo.gif)

## Contact

Questions? Suggestions? Open an issue or reach out!

---

**Built with curiosity, debugged with patience, deployed with pride.**
