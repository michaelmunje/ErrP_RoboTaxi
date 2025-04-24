import torch
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from robotaxi.gameplay.wrappers import make_gymnasium_environment, SimplifiedObservationWrapper, preprocess_observation, preprocess_observation_tamer
import matplotlib.pyplot as plt
from robotaxi.agent.tamer_agent import TAMERAgent
from robotaxi.gameplay.entities import CellType, SnakeDirection, SnakeAction, ALL_SNAKE_ACTIONS
from robotaxi.gameplay.wrappers import preprocess_observation_tamer
from scipy.io import loadmat
import tqdm
import scipy.io
import os
import random

# TAMER Agent Parameters

# w = np.zeros(6)  # Weight vector for 14-feature vector from preprocess_observation
# random initialization of w
np.random.seed(42)
random.seed(42)
w = np.random.uniform(-1, 1, 6)
# w[0] = 0
# w[1] = 0
# w = np.array([-5, 5, -0.1, 0.1, 0.5, -0.5]) # oracle
alpha = 0.01    # Learning rate
max_steps = int(1e4)  # Maximum training steps (adjust as needed)
# max_steps = int(25)  # Maximum training steps (adjust as needed)

# History queues for delayed feedback (s_{t-2}, s_{t-1}, s_t), etc.
s_history = deque(maxlen=3)  # States
a_history = deque(maxlen=3)  # Actions
f_history = deque(maxlen=3)  # Feature vectors

# Feedback counters
positive_feedback_count = 0
neutral_feedback_count = 0
negative_feedback_count = 0
total_feedback_count = 0

# Environment and PPO Model Setup
def make_env(config_filename):
    def _init():
        env = make_gymnasium_environment(config_filename)
        # env = SimplifiedObservationWrapper(env)
        return env
    return _init

env = make_env("./robotaxi/levels/8x8-blank.json")()

def get_feature_vector_tamer(state):
    """Extract 14-element feature vector from state."""
    return preprocess_observation_tamer(state)

def parse_trigger_order_mat(filepath):
    data = loadmat(filepath)
    trigger_order = data['trigger_order']  # shape: (N, 3)

    episodes = {}

    for ep_idx, timestamp, reward in trigger_order:
        ep_idx = int(ep_idx)
        if ep_idx not in episodes:
            episodes[ep_idx] = {'timestamps': [], 'timestamps_raw': [],'rewards': []}
        episodes[ep_idx]['timestamps_raw'].append(int(timestamp))
        sampling_rate = 512
        episodes[ep_idx]['timestamps'].append(timestamp / sampling_rate)
        episodes[ep_idx]['rewards'].append(int(reward))
        
    # normalize timestamps w.r.t. first
    for ep_idx in episodes.keys():
        timestamps = episodes[ep_idx]['timestamps']
        first_timestamp = timestamps[0]
        episodes[ep_idx]['timestamps'] = [t - first_timestamp for t in timestamps]

    return episodes
    
def determine_action(prev_dir, next_dir):
    """
    Determine action based on direction change.
    prev_dir, next_dir: tuples like (dx, dy)
    Returns: 0 (maintain), 1 (left), or 2 (right)
    """
    # Define direction order for turning logic: up, right, down, left
    dir_order = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

    if prev_dir not in dir_order or next_dir not in dir_order:
        raise ValueError(f"Invalid direction: prev={prev_dir}, next={next_dir}")

    prev_idx = dir_order.index(prev_dir)
    next_idx = dir_order.index(next_dir)

    if next_idx == prev_idx:
        return 0  # maintain direction
    elif (next_idx - prev_idx) % 4 == 1:
        return 2  # right turn
    elif (prev_idx - next_idx) % 4 == 1:
        return 1  # left turn
    else:
        # This handles reverse (180°), which shouldn't happen in snake
        raise ValueError(f"Unexpected 180° turn: prev={prev_dir}, next={next_dir}")
    
def parse_log_folder(folderpath):
    episodes = {}
    
    def state_to_array(state_lines):
        return np.array([[int(ch) for ch in line] for line in state_lines], dtype=int)
    
    for fname in sorted(os.listdir(folderpath)):
        if not fname.endswith('.log'):
            continue
        
        filepath = os.path.join(folderpath, fname)
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith("max_step_limit")]
        
        states = []
        directions = []
        rewards = []

        i = 0
        while i < len(lines):
            if lines[i] == "66666666":
                state = lines[i+1:i+8]
                states.append(state_to_array(state))
                i += 8
                while i < len(lines) and lines[i] != "66666666":
                    if lines[i].startswith("R="):
                        rewards.append(int(lines[i][2:]))
                    elif lines[i].startswith("direction:"):
                        dir_str = lines[i].split(":")[1]
                        direction = tuple(map(int, dir_str.strip("()").split(",")))
                        directions.append(direction)
                    i += 1
            else:
                i += 1
        
        next_states = states[1:]
        actions = [determine_action(directions[i], directions[i+1]) for i in range(len(directions)-1)]

        episodes[fname] = {
            'states': states[:-1],
            'next_states': next_states,
            'directions': directions[:-1],
            'rewards': rewards[:-1],
            'actions': actions
        }

    return episodes

def simulate_transition(state, action):
    return TAMERAgent.simulate_transition(state, action)

def choose_action(s_t, w, env):
    """Select action with highest projected reward (Algorithm 3), using simulate_transition."""
    actions = list(range(env.action_space.n))  # Dynamically get action space
    f_t = get_feature_vector_tamer(s_t)
    best_a = None
    max_rew = -float('inf')
        
    # Simulate transitions for each action using simulate_transition
    action_rew_dict = {}
    for a in ALL_SNAKE_ACTIONS:
        s_next = simulate_transition(s_t, a)
        f_next = get_feature_vector_tamer(s_next)

        delta_f = f_next - f_t
        projected_rew = np.dot(w, delta_f)
        action_rew_dict[a] = projected_rew
        
        if projected_rew > max_rew:
            max_rew = projected_rew
            best_a = a
    
    assert best_a is not None, "No best action found"
    return best_a

def update_reward_model(r, f_prev, f_curr, w, alpha):
    """Update reward model weights and return error (Algorithm 2)."""
    delta_f = f_curr - f_prev              # Δf_{t-1,t-2}
    # make sure that the first two dimensions of delta_f are greater than 0
    delta_f[0] = min(delta_f[0], 0)
    delta_f[1] = min(delta_f[1], 0)
    projected_rew = np.dot(w, delta_f)     # w · Δf
    error = r - projected_rew              # r_{t-2} - projected_rew
    print("f_prev:", f_prev)
    print("f_curr:", f_curr)
    print(f"Projected Reward: {projected_rew}, r: {r})")
    print(f"w (to be updated..): {w}")
    print("============")
    w += alpha * error * delta_f           # w += α * error * Δf
    return w, error                        # Return weights and error

# Training Loop with Metric Tracking
episode_lengths = []          # Store length of each episode
sum_errors = []               # Store sum of errors per episode
current_episode_errors = []   # Collect errors within current episode
step_counter = 0              # Count steps in current episode

s_t, _ = env.reset()
f_t = get_feature_vector_tamer(s_t)
pred_f_next = None

subject_id = 6
trigger_data = parse_trigger_order_mat(f"data/trigger_order_sub{subject_id}.mat")
subject_log_folder = f"data/log_files-selected/subject{subject_id}"

episodes = parse_log_folder(subject_log_folder)

EPISODE_TIME_PER_STEP = 5.0 # 5 seconds per step
FILTER_POSITIVE_REWARDS = True
USE_TRIGGER_REWARDS = True
# need to map episode state indices to trigger rewards
trigger_rewards = {}

episodes_filenames = list(episodes.keys())

for trigger_episode_idx in trigger_data.keys():
    episode_filename = episodes_filenames[trigger_episode_idx-1]
    timestamps = trigger_data[trigger_episode_idx]['timestamps']
    rewards = trigger_data[trigger_episode_idx]['rewards']
    current_trigger_rewards = [0 for _ in range(len(episodes[episode_filename]['states']))]
    states = episodes[episode_filename]['states']
    current_time = 0.0
    current_trigger_idx = 0
    for t in range(len(states)):
        # timestamp for current_trigger_idx should be between t*EPISODE_TIME_PER_STEP and (t+1)*EPISODE_TIME_PER_STEP
        current_trigger_rewards[t] = 1
        if current_trigger_idx < len(timestamps) and timestamps[current_trigger_idx] <= t * EPISODE_TIME_PER_STEP:
            if rewards[current_trigger_idx] == 1:
                current_trigger_rewards[t] = 2
            elif rewards[current_trigger_idx] == 2:
                current_trigger_rewards[t] = -1
            current_trigger_idx += 1
    trigger_rewards[episode_filename] = current_trigger_rewards
    
# filter out positive rewards
if FILTER_POSITIVE_REWARDS:
    for fname in episodes_filenames:
        for t in range(len(trigger_rewards[fname])):
            if trigger_rewards[fname][t] == 2:
                trigger_rewards[fname][t] = 0

for fname in episodes_filenames:
    states, actions, rewards, next_states = episodes[fname]['states'], episodes[fname]['actions'], episodes[fname]['rewards'], episodes[fname]['next_states']
    for t in range(len(states)-1):
        # Choose and execute action
        print(f"============Step {t+1}/{len(states)-1}============\n")
        s_t = states[t]
        a_t = actions[t]
        
        pred_f_next = get_feature_vector_tamer(simulate_transition(s_t, a_t))
        a_history.append(a_t)
        
        s_next = next_states[t]
        f_next = get_feature_vector_tamer(s_next)
        
        feedback = rewards[t]  # Feedback for a_{t-2}
        if USE_TRIGGER_REWARDS:
            feedback = trigger_rewards[fname][t]
        print("feedback:", feedback)
        if feedback != 0:  # Update only for non-neutral feedback
            if not np.allclose(f_next, pred_f_next):
                print(f"cmp f_next: {f_next}, \ncmp pred_f_next: {pred_f_next}")
                print("environment transition is not consistent with the simulated transition, skipping update")
            else:
                w, error = update_reward_model(feedback, f_t, f_next, w, alpha)
                current_episode_errors.append(error)  # Collect error for summing
        
        s_t = s_next
        f_t = f_next
        step_counter += 1  # Increment step counter
        
    s_t = s_next
    f_t = f_next
    step_counter += 1  # Increment step counter

    # Record episode metrics
    episode_lengths.append(step_counter)
    sum_error = sum(current_episode_errors) if current_episode_errors else 0
    sum_errors.append(sum_error)

    # Print progress
    print(f"Episode {len(episode_lengths)}: Length = {step_counter}, Sum of Errors = {sum_error}")

    # Reset for next episode
    current_episode_errors = []
    step_counter = 0

n_episodes_eval = 100
average_reward = 0
total_reward = 0
eval_max_steps = 100

# We already fix seed from before, but do it just in case we change training params
np.random.seed(42)
random.seed(42)
env = make_env("./robotaxi/levels/8x8-blank.json")()

for _ in tqdm.tqdm(range(n_episodes_eval)):
    s_t, _ = env.reset()
    pred_f_next = None
    for t in range(eval_max_steps):
        a_t = choose_action(s_t, w, env)
        pred_f_next = get_feature_vector_tamer(simulate_transition(s_t, a_t))
        s_next, reward, done, truncated, _ = env.step(a_t)
        s_t = s_next
        total_reward += reward
        if t >= eval_max_steps - 1:
            truncated = True
        if done or truncated:  # Episode end or truncated or every 1000 steps
            continue
print(f"EVALUATION: Average reward over {n_episodes_eval} episodes: {total_reward / n_episodes_eval}")
    
# After Training: Plot Results and Save to Files
print("Training completed. Final weights:", w)
np.save("tamer_weights_human_data.npy", w)  # Save weights for future use

# Plot episode lengths
plt.figure(figsize=(10, 5))
plt.plot(episode_lengths, label='Episode Length')
plt.xlabel('Episode Number')
plt.ylabel('Episode Length (Steps)')
plt.title('Episode Length Over Time')
plt.grid(True)
plt.savefig('episode_lengths.png')
plt.close()

# Plot sum of errors
plt.figure(figsize=(10, 5))
plt.plot(sum_errors, label='Sum of Errors', color='orange')
plt.xlabel('Episode Number')
plt.ylabel('Sum of Errors')
plt.title('Sum of Errors Per Episode')
plt.grid(True)
plt.savefig('sum_errors.png')
plt.close()

# Feedback Statistics
if total_feedback_count > 0:
    positive_percentage = (positive_feedback_count / total_feedback_count) * 100
    neutral_percentage = (neutral_feedback_count / total_feedback_count) * 100
    negative_percentage = (negative_feedback_count / total_feedback_count) * 100
    print(f"Feedback Statistics:")
    print(f"  Positive: {positive_percentage:.2f}%")
    print(f"  Neutral: {neutral_percentage:.2f}%")
    print(f"  Negative: {negative_percentage:.2f}%")
else:
    print("No feedback was given during training.")

env.close()