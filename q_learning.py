import gymnasium as gym
import numpy as np
import pickle as pkl

# Create CliffWalking environment
env = gym.make("CliffWalking-v1")

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
EPSILON = 0.1   # Exploration rate
ALPHA = 0.1     # Learning rate
GAMMA = 0.9     # Discount factor
NUM_EPISODES = 500

# Epsilon-greedy policy
def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(env.action_space.n)  # random action
    return np.argmax(q_table[state])  # best action

# Training loop
for episode in range(NUM_EPISODES):
    state, info = env.reset()
    done = False
    total_reward = 0
    episode_length = 0

    while not done:
        # Select an action
        action = policy(state, EPSILON)

        # Take step in the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-Learning update
        q_table[state][action] += ALPHA * (
            reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action]
        )

        # Move to next state
        state = next_state
        total_reward += reward
        episode_length += 1

    print(f"Episode: {episode+1}, Length: {episode_length}, Total Reward: {total_reward}")

# Close environment
env.close()

# Save learned Q-table
pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
