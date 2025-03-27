""" Provides adapters for other AI/RL frameworks, such as OpenAI Gym. """

import gymnasium as gym
from gymnasium import spaces
import json
import numpy as np

from .entities import ALL_SNAKE_ACTIONS
from .environment import Environment


class OpenAIGymEnvAdapter(object):
    """ Converts the Snake environment to OpenAI Gym environment format. """

    def __init__(self, env, action_space, observation_space):
        self.env = env
        self.action_space = OpenAIGymActionSpaceAdapter(action_space)
        self.observation_space = np.array(observation_space)

    def seed(self, value):
        self.env.seed(value)

    def reset(self):
        tsr = self.env.new_episode()
        return tsr.observation

    def step(self, action):
        self.env.choose_action(action)
        timestep_result = self.env.timestep()
        tsr = timestep_result
        return tsr.observation, tsr.reward, tsr.is_episode_end, {}


class OpenAIGymActionSpaceAdapter(object):
    """ Converts the Snake action space to OpenAI Gym action space format. """

    def __init__(self, actions):
        self.actions = np.array(actions)
        self.shape = self.actions.shape
        self.n = len(self.actions)

    def sample(self):
        return np.random.choice(self.actions)


def make_openai_gym_environment(config_filename):
    """
    Create an OpenAI Gym environment for the Snake game.
    
    Args:
        config_filename: JSON config for the Snake game level.

    Returns:
        An instance of OpenAI Gym environment.
    """

    with open(config_filename) as cfg:
        env_config = json.load(cfg)

    env_raw = Environment(config=env_config, verbose=1)
    env = OpenAIGymEnvAdapter(env_raw, ALL_SNAKE_ACTIONS, np.zeros((10, 10)))
    return env

class GymnasiumEnvAdapter(gym.Env):
    """Converts the Snake environment to the Gymnasium environment format."""

    def __init__(self, config_filename):
        """
        Initialize the environment with a configuration file.

        Args:
            config_filename (str): Path to the JSON configuration file for the snake game level.
        """
        # Load the configuration and initialize the raw environment
        with open(config_filename) as cfg:
            env_config = json.load(cfg)
        self.env = Environment(config=env_config, verbose=1)

        # Define the observation space (2D grid of integers)
        height = env_config['field'].__len__()
        width = env_config['field'][0].__len__()
        max_value = 6  # Maximum value in the grid (adjust based on game logic)
        self.observation_space = spaces.Box(
            low=0, high=max_value, shape=(height, width), dtype=np.int32
        )

        # Define the action space (discrete actions)
        self.action_space = spaces.Discrete(len(ALL_SNAKE_ACTIONS))
        self.all_actions = ALL_SNAKE_ACTIONS  # Store action objects for mapping

    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.

        Args:
            seed (int, optional): Seed for the random number generator.
            options (dict, optional): Additional options (unused here).

        Returns:
            tuple: (observation, info) where observation is a NumPy array and info is a dict.
        """
        if seed is not None:
            self.env.seed(seed)
        tsr = self.env.new_episode()
        observation = np.array(tsr.observation)  # Ensure observation is a NumPy array
        info = {}  # No additional info provided in this implementation
        return observation, info

    def step(self, action):
        """
        Take a step in the environment using the given action.

        Args:
            action (int): Index of the action to take (0 to len(ALL_SNAKE_ACTIONS)-1).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the integer action to the corresponding action object
        action_obj = self.all_actions[action]
        self.env.choose_action(action_obj)
        tsr = self.env.timestep()

        observation = np.array(tsr.observation)  # Ensure observation is a NumPy array
        reward = tsr.reward
        terminated = tsr.is_episode_end  # Episode ends (e.g., snake dies)
        truncated = False  # No truncation condition in this environment
        info = {}  # No additional info provided

        return observation, reward, terminated, truncated, info

    def close(self):
        """Close the environment and clean up resources."""
        pass  # No specific cleanup required in this case

def make_gymnasium_environment(config_filename):
    """
    Create a Gymnasium environment for the Snake game.

    Args:
        config_filename (str): Path to the JSON configuration file for the snake game level.

    Returns:
        GymnasiumEnvAdapter: An instance of the Gymnasium environment.
    """
    return GymnasiumEnvAdapter(config_filename)


# def preprocess_observation(grid):
#     """Transform a 2D grid into a flat feature vector for the PPO agent."""
#     head_pos = np.where(grid == 4)
#     if len(head_pos[0]) == 0:
#         raise ValueError("Snake head (4) not found in grid")
#     head_x, head_y = head_pos[0][0], head_pos[1][0]

#     body_pos = np.where(grid == 5)
#     if len(body_pos[0]) == 0:
#         direction = 0  # North (initial assumption)
#         body_x, body_y = head_x, head_y
#     else:
#         body_x, body_y = body_pos[0][0], body_pos[1][0]
#         dx, dy = head_x - body_x, head_y - body_y
#         if dx == -1: direction = 0  # North
#         elif dy == 1: direction = 1  # East
#         elif dx == 1: direction = 2  # South
#         elif dy == -1: direction = 3  # West
#         else: direction = 0

#     if direction == 0: next_x, next_y = head_x - 1, head_y
#     elif direction == 1: next_x, next_y = head_x, head_y + 1
#     elif direction == 2: next_x, next_y = head_x + 1, head_y
#     else: next_x, next_y = head_x, head_y - 1
    
#     if 0 <= next_x < grid.shape[0] and 0 <= next_y < grid.shape[1]:
#         next_cell = grid[next_x, next_y]
#         next_hit = next_cell if next_cell in [0, 1, 3, 6] else 0
#     else:
#         next_hit = 6

#     def find_nearest_targets(value):
#         positions = np.where(grid == value)
#         coords = list(zip(positions[0], positions[1]))
#         if len(coords) == 0:
#             return (-1, -1), (-1, -1)  # No targets
#         distances = [abs(head_x - x) + abs(head_y - y) for x, y in coords]
#         sorted_indices = np.argsort(distances)
#         nearest = coords[sorted_indices[0]] if len(coords) > 0 else (-1, -1)
#         second_nearest = coords[sorted_indices[1]] if len(coords) > 1 else (-1, -1)
#         return nearest, second_nearest

#     nearest_1, second_nearest_1 = find_nearest_targets(1)
#     rel_pos_1 = (nearest_1[0] - head_x, nearest_1[1] - head_y) if nearest_1 != (-1, -1) else (-1, -1)
#     nearest_3, second_nearest_3 = find_nearest_targets(3)
#     rel_pos_3 = (nearest_3[0] - head_x, nearest_3[1] - head_y) if nearest_3 != (-1, -1) else (-1, -1)
#     self_coord = (head_x, head_y)
#     facing = direction

#     observation = np.array([
#         next_hit,
#         rel_pos_1[0], rel_pos_1[1],
#         rel_pos_3[0], rel_pos_3[1],
#         self_coord[0], self_coord[1],
#         facing,
#         nearest_1[0], nearest_1[1],
#         second_nearest_1[0], second_nearest_1[1],
#         nearest_3[0], nearest_3[1],
#         second_nearest_3[0], second_nearest_3[1]
#     ], dtype=np.float32)

#     return observation

def preprocess_observation_v1(grid, last_action=None):
    """Transform a 2D grid into a minimal 7-feature vector for the snake."""
    head_pos = np.where(grid == 4)
    if len(head_pos[0]) == 0:
        raise ValueError("Snake head (4) not found in grid")
    head_x, head_y = head_pos[0][0], head_pos[1][0]

    body_pos = np.where(grid == 5)
    if len(body_pos[0]) == 0:
        direction = 0  # North (initial assumption)
        body_x, body_y = head_x, head_y
    else:
        body_x, body_y = body_pos[0][0], body_pos[1][0]
        dx, dy = head_x - body_x, head_y - body_y
        if dx == -1: direction = 0  # North
        elif dy == 1: direction = 1  # East
        elif dx == 1: direction = 2  # South
        elif dy == -1: direction = 3  # West
        else: direction = 0

    if direction == 0: next_x, next_y = head_x - 1, head_y
    elif direction == 1: next_x, next_y = head_x, head_y + 1
    elif direction == 2: next_x, next_y = head_x + 1, head_y
    else: next_x, next_y = head_x, head_y - 1
    
    if 0 <= next_x < grid.shape[0] and 0 <= next_y < grid.shape[1]:
        next_cell = grid[next_x, next_y]
        next_hit = next_cell if next_cell in [0, 1, 3, 6] else 0
    else:
        next_hit = 6

    def find_nearest_target(value):
        positions = np.where(grid == value)
        coords = list(zip(positions[0], positions[1]))
        if len(coords) == 0:
            return (-1, -1)
        distances = [abs(head_x - x) + abs(head_y - y) for x, y in coords]
        nearest_idx = np.argmin(distances)
        return coords[nearest_idx]

    nearest_1 = find_nearest_target(1)
    rel_pos_1 = (nearest_1[0] - head_x, nearest_1[1] - head_y) if nearest_1 != (-1, -1) else (-1, -1)
    nearest_3 = find_nearest_target(3)
    rel_pos_3 = (nearest_3[0] - head_x, nearest_3[1] - head_y) if nearest_3 != (-1, -1) else (-1, -1)

    # Track if last action was forward (0 if None or turn, 1 if forward)
    has_moved_forward = 1 if last_action == 0 else 0  # Assuming 0 is MAINTAIN_DIRECTION

    observation = np.array([
        direction,       # 0: Facing direction
        next_hit,        # 1: Next cell if moving forward
        rel_pos_1[0], rel_pos_1[1],  # 2, 3: Relative direction to nearest 1
        rel_pos_3[0], rel_pos_3[1],  # 4, 5: Relative direction to nearest 3
        has_moved_forward  # 6: Has moved forward recently
    ], dtype=np.float32)

    return observation

def preprocess_observation_v2(grid, last_action=None):
    """Transform a 2D grid into a minimal 7-feature vector for the snake, with relative positions oriented to the snake's facing direction."""
    # Call the original preprocess_observation to reuse its logic
    direction, next_hit, rel_pos_1_0, rel_pos_1_1, rel_pos_3_0, rel_pos_3_1, has_moved_forward = preprocess_observation_v1(grid, last_action)

    rel_pos_1 = (rel_pos_1_0, rel_pos_1_1)
    rel_pos_3 = (rel_pos_3_0, rel_pos_3_1)
    
    # Rotate relative positions based on the snake's facing direction
    def rotate_relative_position(rel_pos, facing):
        if rel_pos == (-1, -1):  # No target, no rotation needed
            return rel_pos
        dx, dy = rel_pos
        if facing == 0:  # North (no rotation)
            return (dx, dy)
        elif facing == 1:  # East (rotate 90° clockwise)
            return (-dy, dx)
        elif facing == 2:  # South (rotate 180°)
            return (-dx, -dy)
        else:  # West (rotate 270° or -90° clockwise)
            return (dy, -dx)

    # Apply rotation to relative positions
    rel_pos_1_rotated = rotate_relative_position(rel_pos_1, direction)
    rel_pos_3_rotated = rotate_relative_position(rel_pos_3, direction)

    # Construct the observation with rotated relative positions
    observation = np.array([
        direction,          # 0: Facing direction
        next_hit,           # 1: Next cell if moving forward
        rel_pos_1_rotated[0], rel_pos_1_rotated[1],  # 2, 3: Relative direction to nearest 1 (rotated)
        rel_pos_3_rotated[0], rel_pos_3_rotated[1],  # 4, 5: Relative direction to nearest 3 (rotated)
        has_moved_forward    # 6: Has moved forward recently
    ], dtype=np.float32)

    return observation

def preprocess_observation(grid, last_action=None):
    return preprocess_observation_v2(grid, last_action)

# # this function featurizes the observation for the TAMER agent, so instead of using positive and negative distance for nearest 1 and 3,
# # use 2 positive value for each original feature, one for positive and one for negative distance. 
# # for instance (1,1) _> (1,0,1,0) and (-1,-1) -> (0,1,0,1)
# def preprocess_observation_tamer(grid, last_action=None):
#     direction, next_hit, rel_pos_1_0, rel_pos_1_1, rel_pos_3_0, rel_pos_3_1, has_moved_forward = preprocess_observation_v2(grid, last_action)
#     rel_pos_1_0_pos = rel_pos_1_0 if rel_pos_1_0 > 0 else 0
#     rel_pos_1_0_neg = rel_pos_1_0 if rel_pos_1_0 < 0 else 0
#     rel_pos_1_1_pos = rel_pos_1_1 if rel_pos_1_1 > 0 else 0
#     rel_pos_1_1_neg = rel_pos_1_1 if rel_pos_1_1 < 0 else 0
#     rel_pos_3_0_pos = rel_pos_3_0 if rel_pos_3_0 > 0 else 0
#     rel_pos_3_0_neg = rel_pos_3_0 if rel_pos_3_0 < 0 else 0
#     rel_pos_3_1_pos = rel_pos_3_1 if rel_pos_3_1 > 0 else 0
#     rel_pos_3_1_neg = rel_pos_3_1 if rel_pos_3_1 < 0 else 0
#     next_hit_1 = 1 if next_hit == 1 else 0
#     next_hit_3 = 1 if next_hit == 3 else 0
#     next_hit_6 = 1 if next_hit == 6 else 0
#     next_hit_0 = 1 if next_hit == 0 else 0
#     observation = np.array([
#         direction,          # 0: Facing direction
#         next_hit_1, 
#         next_hit_3,
#         next_hit_6,
#         next_hit_0,
#         rel_pos_1_0_pos, rel_pos_1_0_neg, rel_pos_1_1_pos, rel_pos_1_1_neg,  # 2, 3, 4, 5: Relative direction to nearest 1
#         rel_pos_3_0_pos, rel_pos_3_0_neg, rel_pos_3_1_pos, rel_pos_3_1_neg,  # 6, 7, 8, 9: Relative direction to nearest 3
#         has_moved_forward    # 10: Has moved forward recently
#     ], dtype=np.float32)
#     return observation

import numpy as np

def preprocess_observation_tamer(grid):
    """Transform a 2D 8x8 grid into a 6-feature vector for the TAMER agent, using a coordinate system where:
    - Rows (x) increase downward (South).
    - Columns (y) increase rightward (East).
    - North: Decrease y (up in columns).
    - East: Increase x (right in rows).
    """
    head_pos = np.where(grid == 4)
    if len(head_pos[0]) == 0:
        raise ValueError("Snake head (4) not found in grid")
    head_x, head_y = head_pos[0][0], head_pos[1][0]  # head_x is row (downward), head_y is column (rightward)

    # 1. Number of 1s (positive targets)
    num_1s = np.sum(grid == 1)

    # 2. Number of 3s (negative targets)
    num_3s = np.sum(grid == 3)

    # 3. Manhattan distance to closest 1
    ones = np.where(grid == 1)
    if len(ones[0]) == 0:
        dist_1 = -1
    else:
        coords_1 = list(zip(ones[0], ones[1]))
        dist_1 = min(abs(head_x - x) + abs(head_y - y) for x, y in coords_1)

    # 4. Manhattan distance to closest 3
    threes = np.where(grid == 3)
    if len(threes[0]) == 0:
        dist_3 = -1
    else:
        coords_3 = list(zip(threes[0], threes[1]))
        dist_3 = min(abs(head_x - x) + abs(head_y - y) for x, y in coords_3)

    # Infer direction from head and body for features 5 and 6
    body_pos = np.where(grid == 5)
    assert len(body_pos[0]) == 1, "Only One Body Allowed"
    body_x, body_y = body_pos[0][0], body_pos[1][0]
    dx, dy = head_x - body_x, head_y - body_y  # dx (row diff), dy (col diff)
    if dx == -1: 
        direction = 0 # North
    elif dy == 1: 
        direction = 1 # East
    elif dx == 1: 
        direction = 2 # South
    elif dy == -1: 
        direction = 3 # West
    else:
        assert False, "Invalid direction"

    # Check all cells along the current direction until hitting a wall, out of bounds, or target
    def check_encounterments(start_x, start_y, direction):
        encounter_1 = 0
        encounter_3 = 0
        x, y = start_x, start_y

        while True:
            # Determine next position based on direction
            if direction == 0:  # North
                x -= 1
            elif direction == 1:  # East
                y += 1
            elif direction == 2:  # South
                x += 1
            elif direction == 3:  # West
                y -= 1

            # Check bounds
            if not (0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]):
                break  # Out of bounds

            # Check cell value
            cell = grid[x, y]
            if cell == 6:  # Wall
                break
            elif cell == 1:  # Positive target
                encounter_1 = 1
                break
            elif cell == 3:  # Negative target
                encounter_3 = 1
                break
            # Continue for empty (0), head (4), or body (5) without stopping

        return encounter_1, encounter_3

    # Get encounterments for straight movement (action 0: maintain direction)
    encounter_1, encounter_3 = check_encounterments(head_x, head_y, direction)

    # 5. First encounterment of 1 if going straight with no turns (now extended to all along path)
    # 6. First encounterment of 3 if going straight with no turns (now extended to all along path)

    # Return 6-element feature vector
    return np.array([num_1s, num_3s, dist_1, dist_3, encounter_1, encounter_3], dtype=np.float32)
    return np.array([num_1s, num_3s, dist_1, dist_3, encounter_1, encounter_3], dtype=np.float32)
    

class SimplifiedObservationWrapper(gym.ObservationWrapper):
    """Wraps the GymnasiumEnvAdapter to a minimal 7-feature observation space."""
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None  # Track last action per environment
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -8, -8, -8, -8, 0]),
            high=np.array([3, 6, 8, 8, 8, 8, 1]),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_action = None
        return preprocess_observation(obs, self.last_action), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = preprocess_observation(obs, self.last_action)
        self.last_action = action  # Update last action
        return processed_obs, reward, terminated, truncated, info

    def observation(self, observation):
        return preprocess_observation(observation, self.last_action)

# class SimplifiedObservationWrapper(gym.ObservationWrapper):
#     """Wraps the GymnasiumEnvAdapter to simplify the observation space to 16 features."""
#     def __init__(self, env):
#         super().__init__(env)
#         # Corrected to 16 features
#         self.observation_space = spaces.Box(
#             low=np.array([0, -8, -8, -8, -8, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]),
#             high=np.array([6, 8, 8, 8, 8, 7, 7, 3, 7, 7, 7, 7, 7, 7, 7, 7]),
#             dtype=np.float32
#         )

#     def observation(self, observation):
#         return preprocess_observation(observation)

# ALL_SNAKE_ACTIONS = [
#     SnakeAction.MAINTAIN_DIRECTION,
#     SnakeAction.TURN_LEFT,
#     SnakeAction.TURN_RIGHT,
# ]