# Kuiper Belt Escape Using Reinforcement Learning

## Aim
The **Kuiper Belt Escape** project aims to explore and navigate two distinct environments—**Frozen Lake** and **MiniGrid Empty**—using various reinforcement learning algorithms. This project serves as a platform for testing the effectiveness of different RL techniques in handling challenges such as sparse rewards, exploration, and strategic decision-making.

## Requirements
To run the environments successfully, ensure you have the following libraries installed:

- **gymnasium**: A toolkit for developing and comparing reinforcement learning algorithms, providing a wide range of environments.
- **numpy**: A powerful numerical computation library that supports large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.
- **matplotlib**: A plotting library for Python that enables the creation of static, animated, and interactive visualizations, useful for visualizing training and testing results.

You can install these libraries using the following command:

```bash
pip install gymnasium numpy matplotlib
```
## Frozen Lake Environment

### Description
The **Frozen Lake** environment simulates a grid world scenario in which an agent must navigate across a frozen surface from a starting position to a goal. The challenge arises from the presence of holes in the ice and the slippery nature of the lake, which may cause the agent to slip and move in unintended directions. This environment tests the agent's ability to learn navigation strategies that minimize risk while maximizing reward.

The layout consists of a **4x4 grid**, where certain tiles are designated as holes. Players must avoid these hazards while making their way to the goal, which is located at the far end of the grid.

![Frozen Lake](../../../_images/frozen_lake.gif)

### State Space
The state space is composed of **16 discrete states** that represent the possible positions of the agent on the 4x4 grid. Each state corresponds to a specific location, allowing the agent to understand its current position relative to both the goal and the holes:

- **0**: Starting position \([0, 0]\)
- **15**: Goal position \([3, 3]\)

The finite nature of the state space simplifies the learning process while providing enough complexity to pose challenges.

### Action Space
The action space consists of **4 discrete actions** that the agent can take:

	LEFT   - 0
	DOWN - 1
	RIGHT  - 2
	UP  - 3


Players must strategically choose their actions to navigate towards the goal while avoiding falling into any holes, creating a need for careful planning and decision-making.

### Reward Function
The reward function is designed to reinforce desirable behavior and discourage risky actions:

- **Reach Goal**: +1 point for successfully arriving at the goal.
- **Fall into Hole**: 0 points, effectively ending the episode without reward.
- **Walk on Frozen Surface**: 0 points, providing no positive reinforcement for simply moving.

This sparse reward structure encourages the agent to focus on finding the most efficient path to the goal.

### Termination Conditions
An episode can terminate under the following conditions:

- The agent successfully reaches the goal, marking a successful navigation.
- The agent falls into a hole, which results in immediate failure.
- The episode reaches a maximum number of steps (100 for the 4x4 grid), enforcing a limit on the exploration time.

### About the Algorithm
The Frozen Lake environment can be effectively solved using reinforcement learning techniques such as:

- **Deterministic Policy Iteration**: A technique that evaluates a fixed policy and improves it iteratively until convergence.
- **Value Iteration**: A dynamic programming method that calculates the optimal value function for each state, helping the agent derive the best possible actions.

These methods allow the agent to learn from its experiences and develop strategies to navigate the environment successfully.

### Results
#### Training Results
![Training Results](path/to/training_graph.png)

#### Testing Results
![Testing Results](path/to/testing_graph.png)

#### Gameplay Demonstration
![Gameplay GIF](../../../_images/frozen_lake_play.gif)

---

## MiniGrid Empty Environment

### Description
The **MiniGrid Empty** environment features an empty room where the agent's primary objective is to reach the green goal square. This environment is particularly beneficial for testing reinforcement learning algorithms in scenarios characterized by sparse rewards and encourages the agent to explore its surroundings effectively.

Agents can either start in a fixed corner or at a random position, depending on the configuration chosen. This variability introduces new challenges for the learning process and enhances the robustness of the algorithms being tested.

![MiniGrid Empty](path/to/minigrid_empty_image.png)

### State Space
The observation space is structured as a dictionary that provides critical information about the agent's current state:

- **Direction**: Represents the agent’s facing direction (Discrete(4)).
- **Image**: A 7x7 RGB image that visually represents the environment (Box(0, 255, (7, 7, 3), uint8)).
- **Mission**: A textual description of the task at hand, such as “Get to the green goal square,” providing context for the agent’s objectives.

This structured observation space aids the agent in understanding its environment and making informed decisions.

### Action Space
The action space comprises **7 discrete actions** that the agent can perform:

| Num | Name     | Action             |
|-----|----------|--------------------|
| 0   | Left     | Turn left          |
| 1   | Right    | Turn right         |
| 2   | Forward  | Move forward       |


The agent must thoughtfully choose its actions to navigate toward the goal while managing the sparse rewards effectively.

### Reward Function
The reward structure is designed to incentivize successful navigation toward the goal:

A reward of:

\[
1 - 0.9 \times \left(\frac{\text{step\_count}}{\text{max\_steps}}\right)
\]

is awarded upon successfully reaching the goal, promoting efficiency. If the agent fails to reach the goal, it receives a reward of \(0\).

### Termination Conditions
An episode can terminate under the following conditions:

- The agent reaches the green goal square, signifying success.
- The episode times out after reaching a predefined maximum number of steps, enforcing a limit on the exploration duration.

### About the Algorithms
The MiniGrid Empty environment can be effectively tackled using several reinforcement learning algorithms, including:

- **Monte Carlo (Every-Visit Variation)**: This approach estimates the value of actions based on complete episodes, helping the agent learn from its experiences.
- **Q-Learning**: A widely-used model-free algorithm that learns action-value pairs (Q-values), enabling the agent to derive an optimal policy through exploration.
- **SARSA (State-Action-Reward-State-Action)**: An on-policy algorithm that updates the value of the current action based on the action taken and the subsequent state, allowing for more cautious learning.
- **SARSA(λ)**: An extension of SARSA that incorporates eligibility traces, allowing the agent to learn from multiple past experiences for quicker convergence.

### Results
#### Training Results
![Training Results](path/to/training_graph.png)

#### Testing Results
![Testing Results](path/to/testing_graph.png)

#### Gameplay Demonstration
![Gameplay GIF](path/to/minigrid_play.gif)

---
