# Reinforcement Learning on CliffWalking

I coded out SARSA and Q-Learning on the Cliff Walking Problem with tabular action-value function.

##### Cliff Walking Results

![Cliff Walking](https://user-images.githubusercontent.com/53657825/178178405-fe853845-cd5d-4c8f-a679-1d2592ae18b5.gif)


This repository implements **Q-Learning** and **SARSA** algorithms on the **CliffWalking** environment using [Gymnasium](https://gymnasium.farama.org/).
The project demonstrates how **on-policy (SARSA)** and **off-policy (Q-Learning)** temporal-difference learning differ in their behavior and learned policies.

---

## 📌 Problem Definition: CliffWalking Environment

* Gridworld: **4 × 12 grid**
* Start state **(S)** = bottom-left corner
* Goal state **(G)** = bottom-right corner
* Cliff = all squares between **S** and **G** along the bottom row

### Rules:

* Reward = **−1** for each step
* Reward = **−100** if the agent falls into the cliff (agent resets to start)
* Goal: Reach **G** with maximum cumulative reward while avoiding the cliff

<p align="center">
  <img src="https://gymnasium.farama.org/_images/cliffwalking.png" width="500"/>
</p>

---

## 📖 Theoretical Background

### 🔹 Markov Decision Process (MDP)

The environment is modeled as an MDP defined by:
[
\langle S, A, P, R, \gamma \rangle
]

* **S** → finite set of states (48 states for CliffWalking)
* **A** → set of actions (Up, Down, Left, Right)
* **P** → transition probabilities
* **R** → reward function
* **γ (gamma)** → discount factor for future rewards

---

### 🔹 Q-Function

The **action-value function** (Q-function) is defined as:
[
Q^\pi(s,a) = \mathbb{E}*\pi \Big[ \sum*{t=0}^\infty \gamma^t r_{t+1} ; \Big| ; s_0 = s, a_0 = a \Big]
]

It represents the expected return when taking action `a` in state `s` under policy `π`.

---

### 🔹 Temporal Difference (TD) Learning

TD learning updates Q-values using **bootstrapping**:
[
Q(s,a) \leftarrow Q(s,a) + \alpha \big[ \text{Target} - Q(s,a) \big]
]

Where:

* **α** = learning rate
* **Target** depends on the algorithm (SARSA or Q-Learning).

---

### 🔹 SARSA (On-Policy TD Control)

* Uses the **actual next action chosen** by the current ε-greedy policy.
* Update rule:
  [
  Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma Q(s',a') - Q(s,a) \Big]
  ]
* **On-policy** → learns values consistent with the current exploration strategy.
* Learns a **safer path**, avoiding the cliff more.

---

### 🔹 Q-Learning (Off-Policy TD Control)

* Uses the **best possible next action** (`max_a Q[s’,a]`) for updates.
* Update rule:
  [
  Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
  ]
* **Off-policy** → learns the optimal greedy policy regardless of exploration.
* Learns a **risky shortest path**, hugging the cliff.

---

## ⚙️ Implementation

* **Language**: Python 3
* **Libraries**:

  * `gymnasium` → environment
  * `numpy` → Q-table operations
  * `opencv-python` → custom visualization
  * `pickle` → save/load learned Q-tables

### 📝 Files

* `q_learning.py` → Q-Learning training
* `sarsa.py` → SARSA training
* `visualize.py` → OpenCV-based visualization of episodes
* `q_learning_q_table.pkl` / `sarsa_q_table.pkl` → saved models

---

## 📊 Experimental Results

* **Q-Learning**:

  * Learns a **shortest risky path** near the cliff.
  * Optimizes for maximum return but risks falling.

* **SARSA**:

  * Learns a **longer but safer path** avoiding the cliff.
  * Reflects on-policy exploration behavior.

<p align="center">
  <b>Comparison:</b><br>
  Q-Learning → Risky Optimal Policy ⚡ <br>
  SARSA → Safer Conservative Policy 🛡️
</p>

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install gymnasium opencv-python numpy
```

### 2️⃣ Train Q-Learning

```bash
python q_learning.py
```

### 3️⃣ Train SARSA

```bash
python sarsa.py
```

### 4️⃣ Visualize Agent

```bash
python visualize.py
```

---

## 🔮 Future Extensions

* Compare convergence speed between Q-Learning and SARSA.
* Add **Expected SARSA** implementation.
* Extend to **Deep Q-Learning (DQN)**.
* Implement reward shaping for safer exploration.

---

## 📚 References

* [Gymnasium CliffWalking Docs](https://gymnasium.farama.org/environments/toy_text/cliff_walking/)

---

