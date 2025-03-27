import numpy as np
from robotaxi.agent import AgentBase
from robotaxi.gameplay.wrappers import preprocess_observation_tamer
import copy

class TAMERAgent(AgentBase):
    """Represents a robottaxi agent powered by a pre-trained TAMER reward model."""

    def __init__(self, w = None, weights_path="tamer_weights.npy"):
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
        delta_fn = new_fn - old_fn
        return np.dot(w, delta_fn)
    
    def act(self, observation, reward):
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
        actions = [0, 1, 2]  # Hardcoded for robottaxi, assuming 3 actions
        best_as = []
        max_rew = -float('inf')

        for a in actions:
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
            action = np.random.choice(actions)
        else:
            action = np.random.choice(best_as)
            # action = best_as[0]
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
            print("Out of bounds or hit wall")
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