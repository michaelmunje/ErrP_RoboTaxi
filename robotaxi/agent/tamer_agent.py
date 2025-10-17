import numpy as np
from robotaxi.agent import AgentBase
from robotaxi.gameplay.wrappers import preprocess_observation_tamer
import copy
from datetime import datetime
from robotaxi.agent.feedback_utils import UserSignalMixer, DiscreteNegativeOnlySignalMixer
from robotaxi.agent.feedback_utils import FeedbackPreProcessor, TINYMLFeedbackPreProcessor, GPPFeedbackPreProcessor, TinyBernoulliFeedbackPreProcessor, CountBasedFeedbackPreProcessor

from robotaxi.agent.game_feature_utils import compute_delta_features_v4, compute_delta_features_v6


# Utility: reclaim focus for pygame window after other GUI activity
def reclaim_pygame_focus() -> None:
    try:
        import pygame
        import os
        import ctypes
        # Ensure event queue is processed
        try:
            pygame.event.pump()
        except Exception:
            pass
        # Avoid resetting display mode; it can change flags (e.g., OPENGL) and break update()
        # X11-specific raise/focus (Linux)
        try:
            info = pygame.display.get_wm_info()
            display = info.get("display")
            window = info.get("window")
            if display and window and os.name == "posix":
                try:
                    x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
                    x11.XRaiseWindow(display, window)
                    # 1 == RevertToPointerRoot per X11
                    x11.XSetInputFocus(display, window, 1, 0)
                    x11.XFlush(display)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # Pygame not available or environment cannot raise focus; ignore silently
        pass

# initialization w maps
w_map = {
    "w1": np.array([ 0.09894706, -0.01005191, -0.05182143, -0.02420872,  0.03493194, -0.08516584]),
    "w2": np.array([ 0.07768216, -0.09317696, -0.05280239,  0.03182322, -0.02639944, 0.09958436]),
    "w10": np.array([ 0.9894706, -0.1005191, -0.5182143, -0.2420872,  0.3493194, -0.8516584]),
    "w20": np.array([ 0.7768216, -0.9317696, -0.5280239,  0.3182322, -0.2639944, 0.9958436]),
    "w30": np.array([ 0.7768216, -0.9317696, +0.5280239,  -0.3182322, -0.2639944, 0.9958436]),
}

def handle_initial_weights(w):
    if w is None: 
        # initialize w to be flat 6 dimensional vector, between -0.1 and 0.1
        w = np.random.uniform(-0.1, 0.1, 6)
        # fixed starting point 1
        
    # if w is an instance of string
    if isinstance(w, str):
        # if the string is one of w1, w2, map it to the corresponding w
        if w in w_map:
            w_key = w
            w = w_map[w_key]
            print(f"using predefined {w_key} initialized to {w}")
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
        
    return w

class TAMERAgent(AgentBase):
    """Represents a robottaxi agent powered by a pre-trained TAMER reward model."""

    def __init__(self, w = None, weights_path="tamer_weights.npy", epsilon = 0.2, feature_version = "v2"):
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
        self.feature_version = feature_version
        
    def begin_episode(self):
        """Reset the agent for a new episode."""
        self.observation = None
        self.last_action = None

    def project_reward(self, observation, action, w):
        if self.feature_version == "v2":
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
        
        elif self.feature_version == "v3":
            simulated_observation = self.__class__.peek_next_state(observation, action)
            delta_fn = compute_delta_features_v3(observation, simulated_observation)
            return np.dot(w, delta_fn)
        
        elif self.feature_version == "v4":
            simulated_observation = self.__class__.peek_next_state(observation, action)
            delta_fn = compute_delta_features_v4(observation, simulated_observation)
            return np.dot(w, delta_fn)
        
        elif self.feature_version == "v6":
            simulated_observation = self.__class__.peek_next_state(observation, action)
            delta_fn = compute_delta_features_v6(observation, simulated_observation)
            return np.dot(w, delta_fn)
        
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
    
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
    def peek_next_state(cls, state, action):
        grid = copy.deepcopy(state)
        rows, cols = grid.shape

        # Find head (4) and mid/body (5)
        head_pos_list = np.where(grid == 4)
        body_pos_list = np.where(grid == 5)
        assert len(head_pos_list[0]) == 1, "Only One Head Allowed"
        assert len(body_pos_list[0]) == 1, "Only One Body Allowed"
        head_x, head_y = head_pos_list[0][0], head_pos_list[1][0]
        body_x, body_y = body_pos_list[0][0], body_pos_list[1][0]

        # Infer direction from head/body
        dx, dy = head_x - body_x, head_y - body_y
        if dx == -1:
            direction = 0  # North
        elif dy == 1:
            direction = 1  # East
        elif dx == 1:
            direction = 2  # South
        elif dy == -1:
            direction = 3  # West
        else:
            direction = 0  # Default North

        # Apply action to get new direction
        if action == 0:
            new_direction = direction
        elif action == 1:
            new_direction = (direction - 1) % 4
        elif action == 2:
            new_direction = (direction + 1) % 4
        else:
            raise ValueError(f"Invalid action: {action}. Expected 0, 1, or 2.")

        def next_pos_from(dir_code):
            if dir_code == 0:
                return head_x - 1, head_y  # North
            if dir_code == 1:
                return head_x, head_y + 1  # East
            if dir_code == 2:
                return head_x + 1, head_y  # South
            if dir_code == 3:
                return head_x, head_y - 1  # West

        def is_blocked(x, y):
            if not (0 <= x < rows and 0 <= y < cols):
                return True
            return grid[x, y] == 6  # Wall

        # Try intended move first
        nx, ny = next_pos_from(new_direction)
        if not is_blocked(nx, ny):
            grid[nx, ny] = 4
            grid[head_x, head_y] = 5
            grid[body_x, body_y] = 0
            return grid

        # Deterministic redirection (wall warp avoidance)
        redirected = new_direction
        if new_direction in (0, 2):  # Approaching North/South edge
            if head_x == body_x:  # Horizontal alignment
                if head_y - body_y > 0:
                    redirected = 1  # East
                elif head_y - body_y < 0:
                    redirected = 3  # West
                tx, ty = next_pos_from(redirected)
                if is_blocked(tx, ty):
                    # Corner: try South then North
                    for alt in (2, 0):
                        tx, ty = next_pos_from(alt)
                        if not is_blocked(tx, ty):
                            redirected = alt
                            break
            else:  # Vertical alignment
                preferred = 1 if head_y < cols / 2 else 3  # East vs West by board-half
                tx, ty = next_pos_from(preferred)
                if is_blocked(tx, ty):
                    preferred = 3 if preferred == 1 else 1
                redirected = preferred
        else:  # Approaching West/East edge
            if head_y == body_y:  # Vertical alignment
                if head_x - body_x > 0:
                    redirected = 2  # South
                elif head_x - body_x < 0:
                    redirected = 0  # North
                tx, ty = next_pos_from(redirected)
                if is_blocked(tx, ty):
                    # Corner: try East then West
                    for alt in (1, 3):
                        tx, ty = next_pos_from(alt)
                        if not is_blocked(tx, ty):
                            redirected = alt
                            break
            else:  # Horizontal alignment
                preferred = 2 if head_x < rows / 2 else 0  # South vs North by board-half
                tx, ty = next_pos_from(preferred)
                if is_blocked(tx, ty):
                    preferred = 0 if preferred == 2 else 2
                redirected = preferred

        fx, fy = next_pos_from(redirected)
        if is_blocked(fx, fy):
            return grid  # No feasible move

        grid[fx, fy] = 4
        grid[head_x, head_y] = 5
        grid[body_x, body_y] = 0
        return grid

    @classmethod
    def simulate_transition(cls, state, action):
        """
        An older version of the peek_next_state, this should not be used anymore
        
        Simulate the next state based on the current state and action for the snake in the robottaxi grid.
        Assumes an 8x8 grid with values: 0 (empty), 1 (positive target), 3 (negative target), 4 (head), 5 (body), 6 (wall).
        Coordinate system: rows (x) increase downward (South), columns (y) increase rightward (East).
        North: -y (up in columns), East: +x (right in rows), South: +y (down in columns), West: -x (left in rows).
        - If new position hits bounds or wall (6), head and body remain unchanged.
        - If head encounters 1 or 3, overwrites the cell with head (4), leaving reward/penalty handling implicit.
        """
        raise DeprecationWarning("simulate_transition is deprecated, please use peek_next_state instead")
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
    Represent a robottaxi agent that is initialized with a set of weights (could be any initialization), 
    but then updates the weights according user feedbacks and gradient descent/ascent.
    Since it is an agent in the game and we want to make it such that it can make improvements, 
    we need to make sure that the agent keep a history of the states and actions, 
    so that it can update the weights accordingly.
    """
    def __init__(self, w = None, save_path = "tamer_weights_online.npy", lr = 0.01, lr_decay = 0.998, feature_version = "v2", no_log = False):
        """ Alpha is the learning rate for the gradient descent/ascent. """
        w = handle_initial_weights(w)
        
        # assert w is a 6 dimensional vector
        assert w.shape == (6,), "w must be a 6 dimensional vector"
        super().__init__(w, save_path, feature_version = feature_version)
        print(f"feature_version in OnlineTAMERAgent.__init__: {feature_version}")
        self.history = [] # history store the (state_0, action_0, reward_0) (state_1, action_1, reward_1) ...
        self.lr = lr
        self.mode = "train"
        self.no_update = False
        # create a logging path to logs/<yyyy-mm-dd>-<hh-mm-ss>-online-tamer.log
        from datetime import datetime
        self.log_path = f"logs/{datetime.now().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}-online-tamer.log"
        self.lr_decay = lr_decay
        if no_log:
            self.log_path = None
        
    def begin_episode(self):
        super().begin_episode()
        # self.history = []
        self.previous_action = None
        self.previous_state = None
    
    def act(self, observation, reward):
        # if self.mode == "train":, than the policy is epsilon-greedy
        # print(f"act called with reward: {reward}")
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
            # print(f"reward appended: {reward}")
            
        self.previous_state = observation
        self.previous_action = action
        # update the tamer weights
        self.update_weights()
        return action
    
    def update_weights(self):
        """
        Update the tamer weights according to the most recent step in history of the states and actions. 
        """
        if self.mode == "eval":
            return
        # print(f"Updating weights with history.. history length: {len(self.history)}")
        if len(self.history) < 3:
            return
        
        delay = 1 
        
        if self.feature_version == "v2":
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
            
        elif self.feature_version == "v3":
            delta_f = compute_delta_features_v3(self.history[-delay-1][0], self.history[-delay][0])
        elif self.feature_version == "v4":
            delta_f = compute_delta_features_v4(self.history[-delay-1][0], self.history[-delay][0])
        elif self.feature_version == "v6":
            delta_f = compute_delta_features_v6(self.history[-delay-1][0], self.history[-delay][0])
        else: 
            raise ValueError(f"Invalid feature version: {self.feature_version}")
        
        
        projected_rew = np.dot(self.w, delta_f)
        user_rew = self.history[-delay-1][2] # r_{t}, which, again, we just added at the current timestep act(*)
        # TODO: User might have a delay in providing the feedback, so we need a way to handler reward delay later 
        
        error = user_rew - projected_rew
        if user_rew == 0:
            # print("user_rew is 0, skipping update")
            return
        
        # update the weights
        update_to_apply = self.lr * error * delta_f
        # multiply the first two weights by 10 # because more signal when the agent is approaching the target, then encounting the target
        # TODO: I suspect that this helps with faster convergence, but I am not sure -- Zhihan
        
        
        self.w += update_to_apply
        # TODO: For debugging, set the last two weights to be 0
        # self.w[-2:] = 0
        self.lr *= self.lr_decay # decay the learning rate
        
        if self.feature_version == "v2":
            print(f"f_prev: {f_prev}, f_curr: {f_curr}")
            print("f_prev:", f_prev)
            print("f_curr:", f_curr)
        if self.feature_version == "v3" or self.feature_version == "v4" or self.feature_version == "v6":
            f_prev = np.zeros(6)
            f_curr = np.zeros(6)
        # log the above information to the log file
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                if self.feature_version == "v2":
                    f.write(f"[FEATURE] f_prev: {f_prev}, f_curr: {f_curr}\n")
                else:
                    f.write(f"[FEATURE] f_prev: {f_prev}, f_curr: {f_curr}\n")
                f.write(f"[REWARD] Projected Reward: {projected_rew}, user_rew: {user_rew}, error: {error}\n")
                # Convert numpy array to string with commas as separators and no newlines
                weight_str = np.array2string(self.w, separator=', ', max_line_width=np.inf)
                f.write(f"[WEIGHT] self.w: {weight_str}\n")
                f.write("============\n")
            
        # write detailed log to the log file
        # in format of (state_t, reward_t), (state_{t+1}, reward_{t+1}), ...
        if self.log_path is not None:
            detailed_log_file = self.log_path.replace(".log", "_detailed.log")
            with open(detailed_log_file, "a") as f:
                history_str_cleaned = str(self.history[-delay-1]).replace('\n', '') # Replace newline with its literal representation
                f.write(f"[state, action, reward] : {history_str_cleaned}\n")
        return
    
    
class OnlineNoisyTAMERAgent(OnlineTAMERAgent):
    """ same as OnlineTAMERAgent, but add noise to feedbacks, controlled by feedback_accuracy"""
    def __init__(self, **kwargs):
        assert 'weight' in kwargs, "weight must be provided"
        assert 'save_path' in kwargs, "save_path must be provided"
        assert 'lr' in kwargs, "lr must be provided (default 0.01)"
        assert 'negative_feedback_only' in kwargs, "negative_feedback_only must be provided (default True)"
        assert 'lr_decay' in kwargs, "lr_decay must be provided (default 0.998)"
        assert 'epsilon_train' in kwargs, "epsilon_train must be provided (default 0.2)"
        assert 'epsilon_test' in kwargs, "epsilon_test must be provided (default 0.1)"
        assert 'feature_version' in kwargs, "feature_version must be provided (default v4)"
        assert 'feedback_tpr' in kwargs, "feedback_tpr must be provided (default 1.0)"
        assert 'feedback_tnr' in kwargs, "feedback_tnr must be provided (default 1.0)"
        assert 'mode' in kwargs, "mode must be provided (default train)"
        assert 'feedback_processor' in kwargs, "feedback_processor must be provided (default None) (eg. 'TINYMLFeedbackPreProcessor')"
        assert 'uncertainty_bonus_scale' in kwargs, "uncertainty_bonus_scale must be provided (default 0.0)"
        
        
        used_keys = ['weight', 'save_path', 'lr', 'negative_feedback_only', 'lr_decay', 'epsilon_train', 'epsilon_test', 'feature_version', 'feedback_tpr', 'feedback_tnr', 'mode', 'feedback_processor', 'uncertainty_bonus_scale']
        # if there is any other key in kwargs, print them out and notice that they are not used
        for key in kwargs:
            if key not in used_keys:
                print(f"Warning: {key}:{kwargs[key]} is not used in OnlineNoisyTAMERAgent")
        
        self.w = handle_initial_weights(kwargs['weight'])
        self.save_path = kwargs['save_path']
        self.lr = kwargs['lr']
        self.negative_feedback_only = kwargs['negative_feedback_only']
        self.lr_decay = kwargs['lr_decay']
        self.epsilon_train = kwargs['epsilon_train']
        self.epsilon_test = kwargs['epsilon_test']
        self.feature_version = kwargs['feature_version']
        self.feedback_tpr = kwargs['feedback_tpr']
        self.feedback_tnr = kwargs['feedback_tnr']
        self.mode = kwargs['mode']
        self.uncertainty_bonus_scale = kwargs['uncertainty_bonus_scale']
        self.margin = kwargs.get('margin', 0.0)
        self.log_path = f"logs/{datetime.now().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}-online-tamer-noisy.log"
        self.log_path_v2 = f"logs/{datetime.now().strftime('%Y-%m-%d')}-{datetime.now().strftime('%H-%M-%S')}-online-tamer-noisy_v2.log"
        self.history = []
        
        self.feedback_signal_mixer = DiscreteNegativeOnlySignalMixer(tpr = self.feedback_tpr, tnr = self.feedback_tnr)
        
        self.feedback_preprocessor = None
        if kwargs['feedback_processor'] == "TINYMLFeedbackPreProcessor":
            self.feedback_preprocessor = TINYMLFeedbackPreProcessor(feature_version = self.feature_version, negative_feedback_only = self.negative_feedback_only)
        elif kwargs['feedback_processor'] == "GPPFeedbackPreProcessor":
            self.feedback_preprocessor = GPPFeedbackPreProcessor(feature_version = self.feature_version, negative_feedback_only = self.negative_feedback_only)
        elif kwargs['feedback_processor'] == "TinyBernoulliFeedbackPreProcessor":
            self.feedback_preprocessor = TinyBernoulliFeedbackPreProcessor(feature_version = self.feature_version, negative_feedback_only = self.negative_feedback_only)
        elif kwargs['feedback_processor'] == "CountBasedFeedbackPreProcessor":
            self.feedback_preprocessor = CountBasedFeedbackPreProcessor(feature_version = self.feature_version, negative_feedback_only = self.negative_feedback_only, margin = self.margin)
        
        # write to log all the parameters
        with open(self.log_path_v2, "a") as f:
            for key in used_keys:
                f.write(f"{key}: {kwargs[key]}\n")
            f.write(f"logfile: {self.log_path_v2}\n")
            f.write(f"w: {self.w}\n")
            f.write("============\n")
    
    
    def log_v2(self, previous_state, current_state, previous_action, reward, noisy_reward, surrogate_reward, weights = [0.0 for _ in range(6)], explored = False):
        with open(self.log_path_v2, "a") as f:
            previous_state_str = str(previous_state).replace('\n', ',')
            current_state_str = str(current_state).replace('\n', ',')
            weights_str = str(weights).replace('\n', ',')
            f.write(f"[previous_state, current_state, previous_action, reward_gt, reward_noisy, reward_surrogate, weights, explored] : {previous_state_str}, {current_state_str}, {previous_action}, {reward}, {noisy_reward}, {surrogate_reward}, {weights_str}, {explored}\n")
    
    def get_uncertainty_bonus(self, prev_state, state):
        # read the history, get the count fo delta 
        pass
    
    def get_ss_transition_count(self, prev_state, state):
        # read the history, compute delta features in history s_{t}, s_{t+1}
        # then count the number of times the delta features are the same
        # return the count
        assert self.feature_version in ["v4", "v6"], "Invalid feature version"
        f = None
        if self.feature_version == "v4":
            f = lambda state, next_state: compute_delta_features_v4(state, next_state)[2:4]
        elif self.feature_version == "v6":
            f = lambda state, next_state: compute_delta_features_v6(state, next_state)[2:4]
        else:
            raise ValueError(f"Invalid feature version: {self.feature_version}")
        delta_features = f(prev_state, state)
        count = 0
        for i in range(len(self.history) - 1):
            h_state = self.history[i][0]
            h_state_next = self.history[i+1][0]
            h_delta_features = f(h_state, h_state_next)
            if np.all(h_delta_features == delta_features):
                count += 1
        return count
        
    
    def act(self, observation, reward):
        
        self.observation = observation
        # ===================== THE ACTING PART, where the action is chosen based on the reward model =====================
        
        # Simulate transitions for each action to find the best
        self.actions = [0, 1, 2]  # Hardcoded for robottaxi, assuming 3 actions
        best_as = []
        max_rew = -float('inf')
        
        candidate_actions = self.actions[::]
        ss_transition_count = []
        action_predicted_rewards = []
        for a in candidate_actions:
            action_predicted_rewards.append(self.project_reward(self.observation, a, self.w))
            next_state = self.__class__.peek_next_state(self.observation, a)
            ss_transition_count.append(self.get_ss_transition_count(self.observation, next_state))
        
        # UCB-style action scoring: reward + beta * sqrt(log(total)/ (count+1))
        counts = np.asarray(ss_transition_count, dtype=float)
        preds = np.asarray(action_predicted_rewards, dtype=float)
        total = float(counts.sum()) + 1.0
        if self.uncertainty_bonus_scale > 0:
            beta = self.uncertainty_bonus_scale
        else:
            beta = 0
        bonus = beta * np.sqrt(np.log(total + 1.0) / (counts + 1.0))
        rew_plus_uncertainty_bonus = preds + bonus
        
        print(f"counts: {counts}")
        print(f"raw predicted rewards: {list(action_predicted_rewards)}")
        print(f"rew_plus_uncertainty_bonus: {rew_plus_uncertainty_bonus}")
        
        for idx, a in enumerate(candidate_actions):
            score = float(rew_plus_uncertainty_bonus[idx])
            if score == max_rew:
                max_rew = score
                best_as.append(a)
            elif score > max_rew:
                max_rew = score
                best_as = [a]
                
        # what are the best action without uncertainty bonus
        best_as_no_bonus = []
        max_rew_no_bonus = -float('inf')
        for idx, a in enumerate(candidate_actions):
            score = float(action_predicted_rewards[idx])
            if score == max_rew_no_bonus:
                max_rew_no_bonus = score
                best_as_no_bonus.append(a)
            elif score > max_rew_no_bonus:
                max_rew_no_bonus = score
                best_as_no_bonus = [a]
        
        explored = False
        # Fallback if no action found (shouldn't happen with valid states)
        if len(best_as) == 0:
            print("Warning: No best action found; choosing randomly")
            action = np.random.choice(self.actions)
        else:
            action = np.random.choice(best_as)
            if action not in best_as_no_bonus:
                print(f"we explored because of the uncertainty bonus")
                explored = True
            # action = best_as[0]
        
            
        if self.mode == "train" and np.random.rand() < self.epsilon_train:
            action = np.random.choice(self.actions)
        elif self.mode == "eval" and np.random.rand() < self.epsilon_test:
            action = np.random.choice(self.actions)
        
        if self.mode not in ["train", "eval"]:
            raise ValueError(f"Invalid mode: {self.mode}, mode must be 'train' or 'eval'")
        
        self.last_action = action
        
        
        
        
        # ===================== ADDING NOISE, this could be either part of the agent or not part of the agent We do not add noise at deployment time =====================
        """
        Mixing the ground truth reward signal
        """
        noisy_reward = reward
        if self.feedback_signal_mixer is not None:
            noisy_reward = self.feedback_signal_mixer.mix_signal(reward)
        
        
        # ===================== HANDLING NOISY FEEDBACK, this is part of the agent =====================
        # by default, if we are not using any feedback preprocessor, then the surrogate reward is the noisy reward
        surrogate_reward = noisy_reward 
        if self.previous_action is not None and self.feedback_preprocessor is not None:
            surrogate_reward = self.feedback_preprocessor.get_feedback(state = self.previous_state, action = self.previous_action, next_state = observation, noisy_reward = noisy_reward)
            if len(self.history) < 50:
                surrogate_reward = noisy_reward # we don't trust the feedback_preprocessor too much when the history is too short
            self.feedback_preprocessor.add_feedback(state = self.previous_state, action = self.previous_action, next_state = observation, noisy_reward = noisy_reward)
            self.feedback_preprocessor.visualize_current_processor(save_path = False, show_plot = True)
            reclaim_pygame_focus()
            
        
        if self.previous_action is None:
            # the is the very first action, so no need to update log and weights, the state t0 and action a0 are save
            # in class variable self.previous_state and self.previous_action, and will be consumed at the next act(*)
            pass 
        else:
            print(f"reward: {reward}, noisy_reward: {noisy_reward}, surrogate_reward: {surrogate_reward}")
            # The reward_t is the reward which will be used by self.update_weights() to update the weights
            self.history.append((self.previous_state, self.previous_action, surrogate_reward, reward)) # state_t, action_t, reward_t, reward_gt) 
            # Note: noisy_reward in the history will be used for weight update, while reward is used for logging
            # print(f"reward appended: {noisy_reward}")
        
            self.log_v2(previous_state = self.previous_state, 
                        current_state = observation,
                        previous_action = self.previous_action, 
                        reward = reward, 
                        noisy_reward = noisy_reward,
                        surrogate_reward = surrogate_reward,
                        weights = self.w,
                        explored = explored)
            
            if self.mode == "train":
                self.update_weights() # this takes reward_t in history, which is the surrogate reward
            
        self.previous_state = observation
        self.previous_action = action
        return action
            