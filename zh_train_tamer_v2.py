import torch
import numpy as np
from collections import deque
from stable_baselines3 import PPO
from robotaxi.gameplay.wrappers import make_gymnasium_environment, SimplifiedObservationWrapper, preprocess_observation, preprocess_observation_tamer
import matplotlib.pyplot as plt
from robotaxi.agent.tamer_agent import TAMERAgent
from robotaxi.gameplay.entities import CellType, SnakeDirection, SnakeAction, ALL_SNAKE_ACTIONS
from robotaxi.gameplay.wrappers import preprocess_observation_tamer

# TAMER Agent Parameters

# w = np.zeros(6)  # Weight vector for 14-feature vector from preprocess_observation
# random initialization of w
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
# ppo_model = PPO.load("ppo_robottaxi_simple.zip", device='cpu')

# # Helper Functions
# def get_feature_vector_ppo(state):
#     """Extract 7-element feature vector from state."""
#     return preprocess_observation(state)

def get_feature_vector_tamer(state):
    """Extract 14-element feature vector from state."""
    return preprocess_observation_tamer(state)

def gen_proxy_evaluator(w):
    def proxy_evaluator(state, action):
        current_f = get_feature_vector_tamer(state)
        
        # for action in ALL_SNAKE_ACTIONS:
        next_state = TAMERAgent.simulate_transition(state, action)
        next_f = get_feature_vector_tamer(next_state)
        delta_f = next_f - current_f
        delta_f[0] = min(delta_f[0], 0)
        delta_f[1] = min(delta_f[1], 0)
        # print("delta_f:", delta_f)
        # print("w:", w)
        # return np.dot(w, delta_f)
        
        # there is a 10% chance the feedback is ramdon from 1,-1,0
        if np.random.rand() < 0.1:
            return np.random.choice([-1, 0, 1])
        if np.dot(w, delta_f) > 1e-4:
            return 1
        elif np.dot(w, delta_f) < -1e-4:
            return -1
        else:
            return 0
    return proxy_evaluator
    # TODO, make this 1, 0, -1 Style
    
w_ = np.array([-5, 5, -0.1, 0.1, 0.5, -0.5])
proxy_evaluator = gen_proxy_evaluator(w_)

# def proxy_evaluator(ppo_model, state, action):
#     """Generate TAMER feedback using PPO policy as oracle."""
#     feat_state = get_feature_vector_ppo(state)
#     state_tensor = torch.tensor(feat_state, dtype=torch.float32).unsqueeze(0)
#     with torch.no_grad():
#         action_dist = ppo_model.policy.get_distribution(state_tensor)
#         probs = action_dist.distribution.probs.numpy()[0]  # [num_actions]
    
#     p_a = probs[action]
#     p_max = np.max(probs)
#     p_min = np.min(probs)
    
#     if p_a == p_max:
#         feedback = 1    # "good" feedback
#     elif p_a == p_min:
#         feedback = 0   # "bad" feedback
#     else:
#         feedback = 0    # "neutral" feedback
    
#     # Update feedback counters
#     global positive_feedback_count, neutral_feedback_count, negative_feedback_count, total_feedback_count
#     if feedback == 1:
#         positive_feedback_count += 1
#     elif feedback == 0:
#         neutral_feedback_count += 1
#     elif feedback == -1:
#         negative_feedback_count += 1
#     total_feedback_count += 1
    
#     return feedback

def simulate_transition(state, action):
    return TAMERAgent.simulate_transition(state, action)

def choose_action(s_t, w, env):
    """Select action with highest projected reward (Algorithm 3), using simulate_transition."""
    actions = list(range(env.action_space.n))  # Dynamically get action space
    f_t = get_feature_vector_tamer(s_t)
    best_a = None
    max_rew = -float('inf')
    
    print("f_prev:", f_t)
    
    # Simulate transitions for each action using simulate_transition
    action_rew_dict = {}
    for a in ALL_SNAKE_ACTIONS:
        s_next = simulate_transition(s_t, a)
        f_next = get_feature_vector_tamer(s_next)
        if a == SnakeAction.MAINTAIN_DIRECTION:
            print("f_up:", f_next)
        if a == SnakeAction.TURN_LEFT:
            print("f_left:", f_next)
        if a == SnakeAction.TURN_RIGHT:
            print("f_right:", f_next)
        delta_f = f_next - f_t
        projected_rew = np.dot(w, delta_f)
        action_rew_dict[a] = projected_rew
        
        if projected_rew > max_rew:
            max_rew = projected_rew
            best_a = a
    print(f"best_a: {best_a}, max_rew: {max_rew}, w: {w}")
    
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
    print(f"w: {w}")
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

for t in range(max_steps):
    # Choose and execute action
    print(f"============Step {t+1}/{max_steps}============\n")
    a_t = choose_action(s_t, w, env)
    
    pred_f_next = get_feature_vector_tamer(simulate_transition(s_t, a_t))
    a_history.append(a_t)
    
    s_next, _, done, truncated, _ = env.step(a_t)
    f_next = get_feature_vector_tamer(s_next)
    
    # print(f"s_t: {s_t}, a_t: {a_t}, s_next: {s_next}")
    # print(f"f_next: {f_next}")
    # Update reward model with delayed feedback (when history is full)
    # print("len(s_history):", len(s_history))
    
    feedback = proxy_evaluator(s_t, a_t)  # Feedback for a_{t-2}
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
    
    if done or truncated:  # Episode end or truncated or every 1000 steps
        # Record episode metrics
        episode_lengths.append(step_counter)
        sum_error = sum(current_episode_errors) if current_episode_errors else 0
        sum_errors.append(sum_error)
        
        # Print progress
        print(f"Episode {len(episode_lengths)}: Length = {step_counter}, Sum of Errors = {sum_error}")
        
        # Reset for next episode
        current_episode_errors = []
        step_counter = 0
        s_t, _ = env.reset()

# After Training: Plot Results and Save to Files
print("Training completed. Final weights:", w)
np.save("tamer_weights.npy", w)  # Save weights for future use

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