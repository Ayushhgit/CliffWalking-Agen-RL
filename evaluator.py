import cv2
import gymnasium as gym
import pickle as pkl
import numpy as np

# Creates an environment
cliffEnv = gym.make("CliffWalking-v1", render_mode=None)

# Loading Q Table
q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))


# Visuals
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    # Goal
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img


# epsilon-greedy policy
def policy(state, explore=0.0):
    if np.random.random() < explore:
        return np.random.randint(cliffEnv.action_space.n)
    return int(np.argmax(q_table[state]))


# Seeing 5 episodes
NUM_EPISODES = 5
for episode in range(NUM_EPISODES):

    # Initializing episode
    state, info = cliffEnv.reset()
    done = False
    total_reward = 0
    episode_length = 0
    frame = initialize_frame()

    while not done:
        # Show the current state to the user
        frame2 = put_agent(frame.copy(), state)
        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)

        # Select action according to policy
        action = policy(state)

        # Take an action in the environment
        next_state, reward, terminated, truncated, info = cliffEnv.step(action)
        done = terminated or truncated

        state = next_state
        episode_length += 1
        total_reward += reward

    print(f"Episode: {episode+1}, Length: {episode_length}, Reward: {total_reward}")

cliffEnv.close()
cv2.destroyAllWindows()
