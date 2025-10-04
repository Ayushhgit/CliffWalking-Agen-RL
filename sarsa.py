import gymnasium as gym
import numpy as np
import pickle as pkl

# Create CliffWalking environment
cliff_env = gym.make("CliffWalking-v1")

# Initialize Q-table: 48 states x 4 actions
q_table = np.zeros((48, 4))

def policy(state, explore=0.0):
    # Greedy action
    action = int(np.argmax(q_table[state]))
    # Exploration (epsilon-greedy)
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4))  # 4 actions
    return action

# Hyperparameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
episodes = 500

for episode in range(episodes):

    done = False
    total_reward = 0
    ep_length = 0

    state, info = cliff_env.reset()
    action = policy(state, EPSILON)

    while not done:
        # Step in environment (Gymnasium API)
        next_state, reward, terminated, truncated, info = cliff_env.step(action)
        done = terminated or truncated

        # SARSA: pick next action from policy
        next_action = policy(next_state, EPSILON)

        # Update Q-value
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][next_action] - q_table[state][action]
        )

        # Move to next state-action
        state = next_state
        action = next_action

        total_reward += reward
        ep_length += 1

    print(f"Episode {episode+1} finished | Steps: {ep_length} | Total Reward: {total_reward}")

cliff_env.close()

# Save Q-table
pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved")
