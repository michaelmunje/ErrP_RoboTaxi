import numpy as np
from stable_baselines3 import PPO
from robotaxi.agent import AgentBase

# class PPOAgent(AgentBase):
#     """Represents a robottaxi agent powered by a pre-trained PPO model."""

#     def __init__(self, model_path="ppo_robottaxi.zip", feature_extractor=None):
#         """
#         Create a new PPO-based agent by loading a pre-trained model.

#         Args:
#             model_path (str): Path to the trained PPO model file (e.g., 'ppo_robottaxi.zip').
#         """
#         # Load the pre-trained PPO model, forcing CPU usage
#         self.model = PPO.load(model_path, device='cpu')
#         self.observation = None  # Current observation for the episode
#         self.feature_extractor = feature_extractor  # Optional feature extractor

#     def begin_episode(self):
#         """Reset the agent for a new episode."""
#         self.observation = None

#     def act(self, observation, reward):
#         """
#         Choose the next action to take based on the current observation.

#         Args:
#             observation: Observable state for the current timestep (e.g., 8x8 grid).
#             reward: Reward received at the beginning of the current timestep (unused here).

#         Returns:
#             The index of the action to take next (e.g., maintain direction, turn left, turn right).
#         """
#         # Update the current observation
#         self.observation = observation
        
#         if self.feature_extractor:
#             # Apply feature extraction if a feature extractor is provided
#             self.observation = self.feature_extractor(observation)

#         # Ensure observation is in the correct format (add batch dimension if needed)
#         obs = np.array(self.observation)
#         if len(obs.shape) == 2:  # Assuming observation is (8, 8), add batch dim
#             obs = np.expand_dims(obs, axis=0)

#         # Predict action using the PPO model (deterministic for inference)
#         action, _ = self.model.predict(obs, deterministic=True)
#         return action.item()  # Convert array/scalar to single integer

#     def get_observation(self):
#         """
#         Get the current observation.

#         Returns:
#             The current observation (e.g., 8x8 grid).
#         """
#         return self.observation

import numpy as np
from stable_baselines3 import PPO
from robotaxi.agent import AgentBase
from robotaxi.gameplay.wrappers import preprocess_observation

class PPOAgent(AgentBase):
    """Represents a robottaxi agent with a minimalistic observation space."""

    def __init__(self, model_path="ppo_robottaxi_minimal.zip", feature_extractor=None):
        self.model = PPO.load(model_path, device='cpu')
        self.observation = None
        self.last_action = None  # Track last action for "has_moved_forward"

    def begin_episode(self):
        self.observation = None
        self.last_action = None

    def act(self, observation, reward):
        self.observation = observation
        # Preprocess with last_action (None at start)
        processed_obs = preprocess_observation(self.observation, self.last_action)
        obs = np.expand_dims(processed_obs, axis=0)
        action, _ = self.model.predict(obs, deterministic=True)
        self.last_action = action.item()  # Update last action
        return self.last_action

    def get_observation(self):
        return self.observation