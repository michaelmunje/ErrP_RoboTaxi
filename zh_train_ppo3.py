from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np
import os

from robotaxi.gameplay.wrappers import make_gymnasium_environment, SimplifiedObservationWrapper

# Custom callback to log rewards every 5,000 steps
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_interval=5000, log_file='zhlog.log'):
        super(RewardLoggerCallback, self).__init__()
        self.log_interval = log_interval
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.last_log_step = 0

    def _on_step(self):
        reward = self.locals['rewards'][0]
        self.current_episode_reward += reward
        done = self.locals['dones'][0]
        if done:
            finish_step = self.num_timesteps
            self.episode_rewards.append((finish_step, self.current_episode_reward))
            self.current_episode_reward = 0

        if self.num_timesteps % self.log_interval == 0:
            episodes_since_last = [
                reward for step, reward in self.episode_rewards
                if step > self.last_log_step and step <= self.num_timesteps
            ]
            with open(self.log_file, 'a') as f:
                if episodes_since_last:
                    avg_reward = np.mean(episodes_since_last)
                    f.write(
                        f"Step {self.num_timesteps}: Average reward over last "
                        f"{len(episodes_since_last)} episodes: {avg_reward:.2f}\n"
                    )
                else:
                    f.write(f"Step {self.num_timesteps}: No episodes completed since last log.\n")
            self.last_log_step = self.num_timesteps
        return True

# Function to create environment instances for parallel execution
def make_env(config_filename):
    def _init():
        env = make_gymnasium_environment(config_filename)
        env = SimplifiedObservationWrapper(env)
        return env
    return _init

def main():
    # Explicitly set device to CPU
    device = 'cpu'
    print(f"Using device: {device}")

    # Set up multiple environments (12 workers for 12 cores)
    config_filename = "./robotaxi/levels/8x8-blank.json"
    n_envs = 12  # Match the number of CPU cores
    vec_env = SubprocVecEnv([make_env(config_filename) for _ in range(n_envs)])

    # Define evaluation environment (consistent with training env)
    eval_env = SubprocVecEnv([make_env(config_filename) for _ in range(1)])  # Single env, but vectorized
    
    obs = vec_env.reset()
    print(f"Observation shape: {obs.shape}")  # Should be (12, 7)

    # Initialize PPO with optimized parameters for short episodes, CPU-only
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=1024,
        batch_size=32,
        n_epochs=10,
        learning_rate=3e-4,
        device=device,
        verbose=1,
        policy_kwargs={"net_arch": [32]}  # Single hidden layer with 32 units
    )

    # Set up callbacks
    reward_logger = RewardLoggerCallback(log_interval=5000, log_file='zhlog.log')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=10000 // n_envs,  # Evaluate every ~10,000 total steps (corrected from 1e5)
        n_eval_episodes=10,
        deterministic=True,
        verbose=1
    )
    callbacks = [reward_logger, eval_callback]

    # Train the model
    total_timesteps = 500000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # Final evaluation
    def evaluate_policy(model, env, n_eval_episodes=10):
        episode_rewards = []
        for _ in range(n_eval_episodes):
            obs = env.reset()  # Vectorized env returns array
            done = np.array([False])
            total_reward = 0
            while not done[0]:
                action, _ = model.predict(obs[0], deterministic=True)  # Unpack first env’s obs
                obs, reward, terminated, truncated = env.step([action])  # Wrap action in list
                total_reward += reward[0]
                done = np.logical_or(terminated, truncated)
            episode_rewards.append(total_reward)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward

    mean_reward, std_reward = evaluate_policy(model, eval_env)
    print(f"Final Evaluation - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the final model
    model.save("ppo_robottaxi_simple")
    print("Model saved as 'ppo_robottaxi_simple.zip'")

    # Clean up
    vec_env.close()
    eval_env.close()

if __name__ == '__main__':
    main()