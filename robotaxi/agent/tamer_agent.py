import numpy as np
from robotaxi.agent import AgentBase
from robotaxi.gameplay.wrappers import preprocess_observation_tamer
import copy
from datetime import datetime


# initialization w maps
w_map = {
    "w1": np.array([ 0.09894706, -0.01005191, -0.05182143, -0.02420872,  0.03493194, -0.08516584]),
    "w2": np.array([ 0.07768216, -0.09317696, -0.05280239,  0.03182322, -0.02639944, 0.09958436]),
}


def handle_collision(f_curr, f_prev):
    if f_curr[0] < f_prev[0]: # crash happened
        f_curr[2] = 0
        f_curr[4] = 0
    if f_curr[1] < f_prev[1]: 
        f_curr[3] = 0
        f_curr[5] = 0
    return f_curr, f_prev

class TAMERAgent(AgentBase):
    """Represents a robottaxi agent powered by a pre-trained TAMER reward model."""

    def __init__(self, w = None, weights_path="tamer_weights.npy", epsilon = 0.2):
        """
        Create a new TAMER-based agent by loading pre-trained weights.

        Args:
            weights_path (str): Path to the saved TAMER weights file (e.g., 'tamer_weights.npy').
        """
        # Load the pre-trained weights
        if w is not None:
            self.w = w
        else:
            self.w = np.load(weights_path)  # Shape: (7,) for 7 features
        self.observation = None  # Current observation for the episode
        self.last_action = None  # Track last action for feature extraction
        self.actions = [0, 1, 2]  # Hardcoded for robottaxi, assuming 3 actions
        self.epsilon = epsilon
        
    def begin_episode(self):
        """Reset the agent for a new episode."""
        self.observation = None
        self.last_action = None

    def project_reward(self, observation, action, w):
        # here's the deal, we are adjusting this to be something obvious,
        # lets say feature number of 1,  # 1, smaller the better
        # number of 3, # 2, bigger the better
        # and manhattan distance to closest 1,  # 3, smaller the better
        # manhattan distance to 3, # 4, bigger the better
        # and the first encounterment is 1 if the snake is going straight with no turns # 5, bigger the better
        # and the first encounterment is 3 if the snake is going straight with no turns # 6, smaller the better
        # (1 is positive reward, 3 is negative reward) 
        # so a w is very obvious, [-0.5, +0.5, -0.1, 0.1, 0.5, -0.5]
        simulated_observation = self.__class__.simulate_transition(observation, action)
        old_fn = preprocess_observation_tamer(observation)
        new_fn = preprocess_observation_tamer(simulated_observation)
        new_fn, old_fn = handle_collision(new_fn, old_fn)
        delta_fn = new_fn - old_fn
        return np.dot(w, delta_fn)
    
    def act(self, observation, reward, epsilon = 0.0):
        """
        Choose the next action based on the TAMER reward model, maximizing projected reward.

        Args:
            observation: Observable state for the current timestep (8x8 grid from raw Environment).
            reward: Reward received at the beginning of the current timestep (unused here).

        Returns:
            The index of the action to take next (e.g., maintain direction, turn left, turn right).
        """
        # Update the current observation (raw 8x8 grid)
        self.observation = observation

        # Preprocess observation into feature vector
        f_t = preprocess_observation_tamer(self.observation)

        # Simulate transitions for each action to find the best
        self.actions = [0, 1, 2]  # Hardcoded for robottaxi, assuming 3 actions
        best_as = []
        max_rew = -float('inf')

        for a in self.actions:
            # Simulate next state (simple assumption, no actual environment step)
            # Use a dummy transition (assumes deterministic movement forward or turn)
            # This is a placeholder; ideally, use a transition model or clone the env
            projected_rew = self.project_reward(self.observation, a, self.w)
            
            # s_next = self.simulate_transition(self.observation, a)
            # f_next = preprocess_observation_tamer(s_next)
            # delta_f = f_next - f_t
            # projected_rew = np.dot(self.w, delta_f)

            if projected_rew == max_rew:
                max_rew = projected_rew
                best_as.append(a)
            elif projected_rew > max_rew:
                max_rew = projected_rew
                best_as = [a]

        # Fallback if no action found (shouldn't happen with valid states)
        if len(best_as) == 0:
            print("Warning: No best action found; choosing randomly")
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(best_as)
            # action = best_as[0]
            
        if np.random.rand() < epsilon:
            action = np.random.choice(self.actions)
        
        self.last_action = action
        return action

    @classmethod
    def simulate_transition(cls, state, action):
        """
        Simulate the next state based on the current state and action for the snake in the robottaxi grid.
        Assumes an 8x8 grid with values: 0 (empty), 1 (positive target), 3 (negative target), 4 (head), 5 (body), 6 (wall).
        Coordinate system: rows (x) increase downward (South), columns (y) increase rightward (East).
        North: -y (up in columns), East: +x (right in rows), South: +y (down in columns), West: -x (left in rows).
        - If new position hits bounds or wall (6), head and body remain unchanged.
        - If head encounters 1 or 3, overwrites the cell with head (4), leaving reward/penalty handling implicit.
        """
        grid = copy.deepcopy(state)  # Copy the grid to avoid modifying the original
        rows, cols = grid.shape  # 8x8 grid

        # Find snake head (4)
        head_pos_list = np.where(grid == 4)
        assert len(head_pos_list[0]) == 1, "Only One Head Allowed"
        head_x, head_y = head_pos_list[0][0], head_pos_list[1][0]

        # Find snake body (5) to infer direction
        body_pos_list = np.where(grid == 5)
        assert len(body_pos_list[0]) == 1, "Only One Body Allowed"
        body_x, body_y = body_pos_list[0][0], body_pos_list[1][0]
        
        # x is row, y is column
        dx, dy = head_x - body_x, head_y - body_y  # dx (row diff), dy (col diff)
        
        if dx == -1: direction = 0  # North
        elif dy == 1: direction = 1  # East
        elif dx == 1: direction = 2  # South
        elif dy == -1: direction = 3  # West
        else: direction = 0  # Default to North
        
        # print(f"Direction: {direction}")

        # Determine new position and direction based on action
        new_direction = direction
        if action == 0:  # Maintain direction (move forward)
            pass  # Keep current direction
        elif action == 1:  # Turn left (e.g., North → West, West → South, etc.)
            new_direction = (direction - 1) % 4
        elif action == 2:  # Turn right (e.g., North → East, East → South, etc.)
            new_direction = (direction + 1) % 4
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0, 1, or 2.")

        # Calculate new head position based on new direction
        if new_direction == 0:  # North 
            new_head_x, new_head_y = head_x - 1, head_y
        elif new_direction == 1:  # East
            new_head_x, new_head_y = head_x, head_y + 1
        elif new_direction == 2:  # South
            new_head_x, new_head_y = head_x + 1, head_y
        elif new_direction == 3:  # West
            new_head_x, new_head_y = head_x, head_y - 1

        # Check bounds and obstacles; if hit, keep head and body unchanged
        if not (0 <= new_head_x < rows and 0 <= new_head_y < cols) or grid[new_head_x, new_head_y] == 6:  # Out of bounds or wall
            asdf = 1
            # print("Out of bounds or hit wall")
            # TODO: Implement wall collision handling
        else:
            grid[new_head_x, new_head_y] = 4 # New head position
            grid[head_x, head_y] = 5 # Old head becomes body
            grid[body_x, body_y] = 0 # Clear old body position
                

        return grid

    def get_observation(self):
        """
        Get the current observation.

        Returns:
            The current observation (8x8 grid).
        """
        return self.observation
    
    
    
class OnlineTAMERAgent(TAMERAgent):
    """
    Represent a robottaxi agent that is initialized with a set of weights (could be any initialization), but then updates the weights according user feedbacks and gradient descent/ascent.
    Since it is an agent in the game and we want to make it such that it can make improvements, we need to make sure that the agent keep a history of the states and actions, so that it can update the weights accordingly.
    """
    def __init__(self, w = None, save_path = "tamer_weights_online.npy", alpha = 0.01, lr_decay = 0.998):
        """ Alpha is the learning rate for the gradient descent/ascent. """
        if w is None: 
            # initialize w to be flat 6 dimensional vector, between -0.1 and 0.1
            w = np.random.uniform(-0.1, 0.1, 6)
            # fixed starting point 1
            
        # if w is an instance of string
        if isinstance(w, str):
            # if the string is one of w1, w2, map it to the corresponding w
            if w in w_map:
                w = w_map[w]
                # now w is a numpy array
            
            # try to parse str to an array with 6 elements
            else:
                try:
                    w = np.array(eval(w))
                    assert w.shape == (6,), "w must be a 6 dimensional vector"
                except:
                    raise ValueError(f"w must be a 6 dimensional vector or one of {w_map.keys()}")
            
        # if w is a list, assert it is 6 dimensional, and convert it to a numpy array
        if isinstance(w, list):
            w = np.array(w)
        
        # assert w is a 6 dimensional vector
        assert w.shape == (6,), "w must be a 6 dimensional vector"
        super().__init__(w, save_path)
        self.history = [] # history store the (state_0, action_0, reward_0) (state_1, action_1, reward_1) ...
        self.alpha = alpha
        self.mode = "train"
        self.no_update = False
        # create a logging path to logs/<yyyy-mm-dd>-<hh-mm-ss>-online-tamer.log
        from datetime import datetime
        self.log_path = f"logs/{datetime.now().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}-online-tamer.log"
        self.lr_decay = lr_decay
        
    def begin_episode(self):
        super().begin_episode()
        self.history = []
        self.previous_action = None
        self.previous_state = None
    
    def act(self, observation, reward):
        # if self.mode == "train":, than the policy is epsilon-greedy
        print(f"act called with reward: {reward}")
        if self.mode == "train":
            action = super().act(observation, reward, epsilon = self.epsilon)
            # epsilon-greedy
        else:
            action = super().act(observation, reward, epsilon = 0.05)
            # some epsilon for loop breaking
        
        if self.previous_state is None: # means this is the first actiona
            pass
        else:
            self.history.append((self.previous_state, self.previous_action, reward)) # state_t, action_t, reward_t)
            print(f"reward appended: {reward}")
            
        self.previous_state = observation
        self.previous_action = action
        # update the tamer weights
        self.update_weights()
        return action
    
    def update_weights(self):
        """
        Update the tamer weights according to the most recent step in history of the states and actions. 
        """
        if self.no_update:
            return
        print(f"Updating weights with history.. history length: {len(self.history)}")
        if len(self.history) < 3:
            return
        
        delay = 1 
        
        f_curr = preprocess_observation_tamer(self.history[-delay][0]) # f(s_{t+1})last entry in the history, which we just added at the current timestep act(*)
        f_prev = preprocess_observation_tamer(self.history[-delay-1][0]) # f(s_{t})
        
        
        

        
        # The below is necessary because some limitations of the underlying game implementation
        f_prev, f_curr = handle_collision(f_prev, f_curr)
        
        # The random creation of new game element could messed up the feature vector
        delta_f = f_curr - f_prev
        
        delta_f[2:4] *= 0.3
        delta_f[:2] *= 5
        
        delta_f[0] = min(delta_f[0], 0)
        delta_f[1] = min(delta_f[1], 0)
        
        
        projected_rew = np.dot(self.w, delta_f)
        user_rew = self.history[-delay-1][2] # r_{t}, which, again, we just added at the current timestep act(*)
        # TODO: User might have a delay in providing the feedback, so we need a way to handler reward delay later 
        
        error = user_rew - projected_rew
        if user_rew == 0:
            print("user_rew is 0, skipping update")
            return
        
        # update the weights
        update_to_apply = self.alpha * error * delta_f
        # multiply the first two weights by 10 # because more signal when the agent is approaching the target, then encounting the target
        # TODO: I suspect that this helps with faster convergence, but I am not sure -- Zhihan
        
        
        self.w += update_to_apply
        # TODO: For debugging, set the last two weights to be 0
        # self.w[-2:] = 0
        self.alpha *= self.lr_decay # decay the learning rate
        
        print(f"f_prev: {f_prev}, f_curr: {f_curr}")
        print("f_prev:", f_prev)
        print("f_curr:", f_curr)
        print(f"Projected Reward: {projected_rew}, user_rew: {user_rew}")
        print(f"w (updated): {self.w}, error: {error}")
        print(f"delta_f: {delta_f}")
        print("============")      # w += α * error * Δf
        # log the above information to the log file
        with open(self.log_path, "a") as f:
            f.write(f"[FEATURE] f_prev: {f_prev}, f_curr: {f_curr}\n")
            f.write(f"[REWARD] Projected Reward: {projected_rew}, user_rew: {user_rew}, error: {error}\n")
            # Convert numpy array to string with commas as separators and no newlines
            weight_str = np.array2string(self.w, separator=', ', max_line_width=np.inf)
            f.write(f"[WEIGHT] self.w: {weight_str}\n")
            f.write("============\n")
            
        # write detailed log to the log file
        # in format of (state_t, reward_t), (state_{t+1}, reward_{t+1}), ...
        detailed_log_file = self.log_path.replace(".log", "_detailed.log")
        with open(detailed_log_file, "a") as f:
            history_str_cleaned = str(self.history[-delay-1]).replace('\n', '') # Replace newline with its literal representation
            f.write(f"[state, action, reward] : {history_str_cleaned}\n")
        return

    
class OnlineNoisyTAMERAgent(OnlineTAMERAgent):
    """ same as OnlineTAMERAgent, but add noise to feedbacks, controlled by feedback_accuracy"""
    def __init__(self, w = None, save_path = "tamer_weights_online_noisy.npy", alpha = 0.01, feedback_accuracy = 1.0, negative_feedback_only = False, lr_decay = 0.998, epsilon_train = 0.2, epsilon_test = 0.1):
        super().__init__(w, save_path, alpha, lr_decay)
        self.feedback_accuracy = feedback_accuracy
        self.log_path = f"logs/{datetime.now().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}-online-tamer-noisy.log"
        self.negative_feedback_only = negative_feedback_only
        self.epsilon_train = epsilon_train
        self.epsilon_test = epsilon_test
        
    # log both the reward and noisy reward (log the noisy reward as normal reward, and log the normal reward as reward_gt)
    
    def act(self, observation, reward):
        # if self.mode == "train":, than the policy is epsilon-greedy
        print(f"act called with reward: {reward}")
        if self.mode == "train":
            # instead of super, call the act method of TAMERAgent
            # the grandparent act calculate the epsilon optimal action and does not do updates
            action = TAMERAgent.act(self, observation, reward, epsilon = self.epsilon_train)
            # epsilon-greedy
        else:
            action = TAMERAgent.act(self, observation, reward, epsilon = self.epsilon_test)
            # some epsilon for loop breaking
        
        if self.previous_state is None: # means this is the first actiona
            pass
        else:
            if np.random.rand() <= self.feedback_accuracy:
                noisy_reward = reward
            else:
                possible_rewards = [0, 1, -1]
                if self.negative_feedback_only:
                    possible_rewards = [0, -1]
                if reward not in possible_rewards:
                    print("="*100 + "\n" + "+++++ WARNING +++++" + "\n" + "="*100)
                    print(f"reward {reward} not in possible_rewards {possible_rewards}, skipping update")
                    print("Are you giving positive feedbacks? while the setting is negative_feedback_only?")
                    print("="*100 + "\n" + "+++++ WARNING +++++" + "\n" + "="*100)  
                    if not self.negative_feedback_only:
                        raise ValueError("reward not in possible_rewards")
                else:
                    possible_rewards.remove(reward)
                noisy_reward = np.random.choice(possible_rewards)
            self.history.append((self.previous_state, self.previous_action, noisy_reward, reward)) # state_t, action_t, reward_t, reward_gt)
            # Note: noisy_reward in the history will be used for weight update, while reward is used for logging
            print(f"reward appended: {noisy_reward}")
            
        self.previous_state = observation
        self.previous_action = action
        # update the tamer weights
        self.update_weights()
        return action
            