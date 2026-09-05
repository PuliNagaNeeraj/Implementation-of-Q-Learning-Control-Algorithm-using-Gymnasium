# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement
To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment. The agent must learn an optimal action-value function through interaction with the environment and select suitable actions to reach the goal while avoiding the holes.


## Software Requirements
* **Programming Language:** Python
* **Library:** Gymnasium
* **Numerical Computation:** NumPy
* **Visualization:** Matplotlib
* **Environment:** FrozenLake-v1
* **Platform:** Jupyter Notebook / Google Colab


## Environment Description

The FrozenLake-v1 environment is a grid-world reinforcement learning environment provided by Gymnasium.

The environment consists of a **4 × 4 grid** containing:

* **S** – Starting state
* **F** – Frozen/safe surface
* **H** – Hole, which terminates the episode
* **G** – Goal state

The agent starts from the initial state and must navigate through the frozen surface to reach the goal without falling into a hole.

The environment contains **16 states** and **4 possible actions**:

| Action | Direction |
| ------ | --------- |
| 0      | Left      |
| 1      | Down      |
| 2      | Right     |
| 3      | Up        |

The agent receives a reward when it successfully reaches the goal. Falling into a hole terminates the episode.


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
Q-Learning Algorithm:

Step 1: Initialize the Q-table with zeros.

Step 2: Set the learning parameters:
        α = 0.1
        γ = 0.93
        ε = 1.0

Step 3: For each training episode:
        a. Reset the environment.
        b. Obtain the initial state.
        c. Select an action using the epsilon-greedy strategy.
        d. Execute the action in the environment.
        e. Observe the next state and reward.
        f. Update the Q-value using the Q-Learning update rule.
        g. Move to the next state.
        h. Repeat until the episode terminates.

Step 4: Decrease epsilon after every episode using epsilon decay.

Step 5: Repeat the training process for 10,000 episodes.

Step 6: Calculate the state-value function by taking the maximum Q-value for every state.

Step 7: Extract the learned policy by selecting the action with the highest Q-value for every state.

Step 8: Calculate the average reward obtained during the last 1,000 episodes.

Step 9: Display the final Q-table, state-value function, learned policy, and average reward.


## Python Program

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------
epsilon = epsilon_start

for episode in range(num_episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps_per_episode):

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])
        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )
        state = next_state

        total_reward += reward
        if terminated or truncated:
            break
            
    episode_rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# -------------------------------------------------
# Extract State-Value Function and Policy
# -------------------------------------------------

state_values = np.max(Q, axis=1)

learned_policy = np.argmax(Q, axis=1)

## Output

```text
Name: Puli Naga Neeraj

Reg No: 212223240130

Final Q-table:
[[0.095 0.092 0.104 0.092]
 [0.069 0.088 0.051 0.094]
 [0.095 0.088 0.092 0.082]
 [0.064 0.04  0.036 0.079]
 [0.131 0.085 0.103 0.071]
 [0.    0.    0.    0.   ]
 [0.092 0.038 0.087 0.037]
 [0.    0.    0.    0.   ]
 [0.123 0.119 0.149 0.179]
 [0.248 0.274 0.173 0.218]
 [0.289 0.28  0.191 0.141]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.203 0.298 0.396 0.28 ]
 [0.489 0.679 0.547 0.489]
 [0.    0.    0.    0.   ]]

Estimated State-Value Function:
[[0.104 0.094 0.095 0.079]
 [0.131 0.    0.092 0.   ]
 [0.179 0.274 0.289 0.   ]
 [0.    0.396 0.679 0.   ]]

Learned Policy:
[['R' 'U' 'L' 'U']
 ['L' 'L' 'L' 'L']
 ['U' 'D' 'L' 'L']
 ['L' 'R' 'D' 'L']]

Average reward over last 1000 episodes: 0.486
```
<img width="865" height="525" alt="image" src="https://github.com/user-attachments/assets/9e3e240f-9e82-4c7a-aa61-362eb649a6eb" />

---

## Result


The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the action-value function through repeated interaction with the environment. The final Q-table, state-value function, and learned policy were obtained successfully. The agent achieved an average reward of approximately 0.486 over the last 1000 training episodes.

---

## Inference

The experiment demonstrates that Q-Learning can learn an effective policy for the FrozenLake environment without requiring a model of the environment.

Initially, the agent explores the environment using a high epsilon value. As training progresses, epsilon decreases and the agent increasingly exploits the learned Q-values.

The final Q-table represents the learned action values for each state-action pair. The state-value function is obtained by selecting the maximum Q-value for each state, while the learned policy selects the action having the highest Q-value.

The obtained average reward shows that the agent successfully learned to reach the goal in a significant number of episodes despite the stochastic nature of the slippery FrozenLake environment.

---

