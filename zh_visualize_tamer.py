import pygame
import numpy as np
import time
import json

from robotaxi.gameplay.environment import Environment
from robotaxi.gameplay.entities import CellType, SnakeDirection, SnakeAction, ALL_SNAKE_ACTIONS
# from robotaxi.agent.ppo_agent import PPOAgent  # Updated import path
from robotaxi.gameplay.wrappers import SimplifiedObservationWrapper, preprocess_observation
from robotaxi.agent.tamer_agent import TAMERAgent  # Updated import path
import imageio.v2 as imageio


class PyGameGUI:
    """Simplified Pygame GUI to visualize a trained PPO agent's policy in Robotaxi."""

    FPS_LIMIT = 60
    TIMESTEP_DELAY = 100  # Milliseconds between steps (adjust for speed)

    def __init__(self, field_size=8):
        pygame.init()
        self.field_size = field_size
        self.cell_size = 96 * 8 // self.field_size  # Scale cell size based on 8x8 grid
        self.screen_size = (self.field_size * self.cell_size, self.field_size * self.cell_size)
        self.screen = pygame.display.set_mode(self.screen_size)
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        self.fps_clock = pygame.time.Clock()
        self.agent = None
        self.env = None
        self.pause = False
        self.timestep_watch = Stopwatch()

        # Load icons (simplified set)
        self.wall_icon = pygame.transform.scale(pygame.image.load("icon/forest.png"), (self.cell_size, self.cell_size))
        self.good_fruit_icon = pygame.transform.scale(pygame.image.load("icon/man.png"), (self.cell_size * 2 // 3, self.cell_size * 2 // 3))
        self.bad_fruit_icon = pygame.transform.scale(pygame.image.load("icon/road_block.png"), (self.cell_size * 2 // 3, self.cell_size * 2 // 3))
        self.lava_icon = pygame.transform.scale(pygame.image.load("icon/purple_car.png"), (self.cell_size, self.cell_size))
        self.reward_icon = pygame.transform.scale(pygame.image.load("icon/dollar.png"), (self.cell_size // 3, self.cell_size // 3))
        
        # Snake icons (simple car scheme)
        self.south = pygame.transform.scale(pygame.image.load("icon/auto_bus_south.png"), (self.cell_size, self.cell_size - 5))
        self.north = pygame.transform.scale(pygame.image.load("icon/auto_bus_north.png"), (self.cell_size, self.cell_size - 5))
        self.east = pygame.transform.scale(pygame.image.load("icon/auto_bus_east.png"), (self.cell_size, self.cell_size - 5))
        self.west = pygame.transform.flip(self.east, True, False)

        self.text_font = pygame.font.Font("fonts/gyparody_hv.ttf", int(23 * (self.cell_size / 40.0)))
        self.num_font = pygame.font.Font("fonts/gyparody_tf.ttf", int(36 * (self.cell_size / 40.0)))
        pygame.display.set_caption('Robotaxi PPO Visualization')

    def load_environment(self, environment):
        """Load the raw robottaxi environment."""
        self.env = environment

    def load_agent(self, agent):
        """Load the PPO agent."""
        self.agent = agent

    def render_scoreboard(self, score, steps_remaining):
        """Render a simple scoreboard with score and steps remaining."""
        text = ("Score", str(score))
        disp_text = self.text_font.render(text[0], True, (0, 0, 0))
        disp_num = self.num_font.render(text[1], True, (0, 0, 0))
        self.screen.blit(disp_text, (self.cell_size // 2, self.screen_size[1] - 2 * self.cell_size))
        self.screen.blit(disp_num, (self.cell_size // 2, self.screen_size[1] - self.cell_size))

    def render_cell(self, x, y):
        """Render a single cell in the grid."""
        cell_coords = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        cell_type = self.env.field[x, y]

        if cell_type == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        elif cell_type == CellType.WALL:
            self.screen.blit(self.wall_icon, cell_coords)
        elif cell_type == CellType.GOOD_FRUIT:
            self.screen.blit(self.good_fruit_icon, (x * self.cell_size + self.cell_size // 6, y * self.cell_size + self.cell_size // 6))
        elif cell_type == CellType.BAD_FRUIT:
            self.screen.blit(self.bad_fruit_icon, (x * self.cell_size + self.cell_size // 6, y * self.cell_size + self.cell_size // 6))
        elif cell_type == CellType.LAVA:
            self.screen.blit(self.lava_icon, cell_coords)
        elif cell_type == CellType.SNAKE_HEAD:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
            if self.env.snake.direction == SnakeDirection.NORTH:
                icon = self.north
            elif self.env.snake.direction == SnakeDirection.SOUTH:
                icon = self.south
            elif self.env.snake.direction == SnakeDirection.EAST:
                icon = self.east
            elif self.env.snake.direction == SnakeDirection.WEST:
                icon = self.west
            self.screen.blit(icon, cell_coords)
        else:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)

    def render(self):
        """Render the entire grid."""
        self.screen.fill(Colors.SCREEN_BACKGROUND)
        for x in range(self.env.field.size):
            for y in range(self.env.field.size):
                self.render_cell(x, y)

    def transition_animation(self, x, y, x0, y0, reward):
        """Simple animation for snake movement."""
        intermediate_frames = 5  # Fewer frames for faster visualization
        for i in range(intermediate_frames):
            self.screen.fill(Colors.SCREEN_BACKGROUND)
            for px in range(self.env.field.size):
                for py in range(self.env.field.size):
                    if (px, py) != (x, y) and (px, py) != (x0, y0):
                        self.render_cell(px, py)
            
            # Interpolate position
            interp_x = x0 + (x - x0) * i / intermediate_frames
            interp_y = y0 + (y - y0) * i / intermediate_frames
            cell_coords = pygame.Rect(
                interp_x * self.cell_size,
                interp_y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            if self.env.snake.direction == SnakeDirection.NORTH:
                icon = self.north
            elif self.env.snake.direction == SnakeDirection.SOUTH:
                icon = self.south
            elif self.env.snake.direction == SnakeDirection.EAST:
                icon = self.east
            elif self.env.snake.direction == SnakeDirection.WEST:
                icon = self.west
            self.screen.blit(icon, cell_coords)

            # Show reward effect
            if i == intermediate_frames - 1 and reward > 0:
                self.screen.blit(self.reward_icon, (x * self.cell_size + self.cell_size // 3, y * self.cell_size + self.cell_size // 3))

            score = self.env.stats.sum_episode_rewards
            steps_remaining = self.env.max_step_limit - self.env.timestep_index
            self.render_scoreboard(score, steps_remaining)
            pygame.display.update()
            self.fps_clock.tick(self.FPS_LIMIT)

    def run(self, num_episodes: int = 1):
        """Visualise the PPO agent and save all episodes to a GIF."""
        MAX_STEPS     = 100      # hard per-episode step cap
        GIF_INTERVAL  = 0.0333333333333333333333333333     # seconds between captured frames
        gif_frames    = []

        self.fps_clock   = pygame.time.Clock()
        last_gif_stamp   = 0.0    # wall-clock time of previous captured frame

        for episode in range(num_episodes):
            timestep        = self.env.new_episode()
            self.agent.begin_episode()
            self.render()
            pygame.display.update()

            # initial frame
            gif_frames.append(
                pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0, 1)
            )
            last_gif_stamp = time.perf_counter()

            running         = True
            last_head       = list(self.env.snake.head)
            episode_reward  = 0.0
            step_counter    = 0

            print(f"Episode {episode + 1} started")

            while running:
                # ---------- handle UI / quit  ----------
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.pause = not self.pause
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                    elif event.type == pygame.QUIT:
                        running = False

                if self.pause:
                    continue

                # ---------- fixed-timestep advance ----------
                if self.timestep_watch.time() >= self.TIMESTEP_DELAY:
                    self.timestep_watch.reset()

                    # agent → env
                    action_idx  = self.agent.act(timestep.observation, timestep.reward)
                    action      = ALL_SNAKE_ACTIONS[action_idx]
                    self.env.choose_action(action)
                    timestep    = self.env.timestep()

                    # bookkeeping
                    episode_reward += timestep.reward
                    step_counter   += 1
                    print(f"Step {self.env.timestep_index}: "
                        f"Cumulative Reward = {episode_reward:.2f}")

                    # render + small head-motion animation
                    curr_head = list(self.env.snake.head)
                    self.transition_animation(
                        curr_head[0], curr_head[1],
                        last_head[0], last_head[1],
                        timestep.reward
                    )
                    last_head = curr_head

                    # ---------- GIF capture (≤ 10 fps) ----------
                    now = time.perf_counter()
                    if now - last_gif_stamp >= GIF_INTERVAL:
                        gif_frames.append(
                            pygame.surfarray.array3d(
                                pygame.display.get_surface()
                            ).swapaxes(0, 1)
                        )
                        last_gif_stamp = now

                    # ---------- termination checks ----------
                    if step_counter >= MAX_STEPS:
                        print(f"Reached step limit ({MAX_STEPS}); ending episode.")
                        running = False

                    if timestep.is_episode_end and running:
                        self.render()
                        self.render_scoreboard(
                            self.env.stats.sum_episode_rewards,
                            self.env.max_step_limit - self.env.timestep_index
                        )
                        pygame.display.update()
                        print(f"Episode {episode + 1} ended with total reward "
                            f"{episode_reward:.2f}")
                        time.sleep(1)
                        running = False

                pygame.display.update()
                self.fps_clock.tick(self.FPS_LIMIT)

        pygame.quit()

        # ---------- write GIF ----------
        if gif_frames:
            imageio.mimsave("snake_episodes.gif", gif_frames, duration=150)   # 1 / 0.10 s
            print("✅  Saved all episodes to snake_episodes.gif")



class Stopwatch:
    def __init__(self):
        self.start_time = pygame.time.get_ticks()
    def reset(self):
        self.start_time = pygame.time.get_ticks()
    def time(self):
        return pygame.time.get_ticks() - self.start_time

class Colors:
    SCREEN_BACKGROUND = (119, 119, 119)
    SCORE = (120, 100, 84)
    SCORE_GOOD = (50, 205, 50)

# Main execution
if __name__ == '__main__':
    # Load the raw environment
    config_filename = "./robotaxi/levels/8x8-blank.json"
    with open(config_filename) as cfg:
        env_config = json.load(cfg)
    import random
    np.random.seed(0)
    random.seed(0)
    env = Environment(config=env_config, verbose=1)

    # # Load the tamer
    # weight_path = "tamer_weights.npy"
    # w = np.load(weight_path)
    
    # oracle
    # w = np.array([-5, 5, -0.1, 0.1, 0.5, -0.5])
    # w = np.array()
    
    # SUBJECT 4
    # FILTER_POSITIVE_REWARDS = True
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 16.3
    w = np.array([-0.27046622,  0.90610038, -0.04634391, -0.12070922, -0.53461466, -0.53443799])
    
    # SUBJECT 4
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 41.76
    # w = np.array([-0.27569902,  0.91025802, -0.05950488, -0.05818661, -0.53211654, -0.52681638])
    
    # SUBJECT 4
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = False
    # EVALUATION: Average reward over 100 episodes: 5.86
    # w = np.array([-0.31478682,  0.90142861,  0.19107897,  0.08656659, -0.76835702, -0.6145178 ])
    
    # SUBJECT 5
    # FILTER_POSITIVE_REWARDS = True
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 15.51
    # w = np.array([-0.27310332  0.85354614 -0.23575854 -0.13191944 -0.47454534 -0.51415956])
    
    # SUBJECT 5
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 16.43
    # w = np.array([[-0.59463831  0.83639959 -0.34378327 -0.13998706 -0.4895685  -0.53164084]])
    
    # SUBJECT 5
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = False
    # EVALUATION: Average reward over 100 episodes: 1.18
    # w = np.array([[-0.16584424  0.9399954  -0.1783882   0.36064267 -0.77822711 -0.41402435]])
    
    # SUBJECT 6
    # FILTER_POSITIVE_REWARDS = True
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 26.43
    # w = np.array([-0.34476199,  0.8966019,  -0.0450154,  -0.08930686, -0.45620399, -0.49967804])
    
    # SUBJECT 6
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = True
    # EVALUATION: Average reward over 100 episodes: 34.94
    # w = np.array([-0.34173444  0.90222936 -0.03953472 -0.04565889 -0.4769405  -0.50025944])
    
    # SUBJECT 6
    # FILTER_POSITIVE_REWARDS = False
    # USE_TRIGGER_REWARDS = False
    # EVALUATION: Average reward over 100 episodes: 5.78
    # w = np.array([-0.25091976  0.78169215  0.4221407   0.5749483  -0.68796272 -0.66827629])
    
    # FILTER POSITIVE REWARDS, GIVE NEUTRAL REWARDS 1 REWARD
    # w = np.array([-0.26848768, -0.14508542, -0.02450691,  0.09520072, -0.08813761,  0.48581424])
    # w = np.array([ 0.44796108, -0.6748461,   0.04378207, -0.03218805, -0.2686715,  -0.24653979])
    # w = np.array([ 0.10870472,  0.79722621, -0.67809553, -0.2042775,  -0.71542436, -0.83479586])
    # w = np.array([-0.1, 0, -0.3, 0.1, 0.5, -0.5])
    agent = TAMERAgent(w=w)
    
    # trained
    # agent = TAMERAgent()
    # agent = PPOAgent(model_path="ppo_robottaxi_simple.zip", feature_extractor=preprocess_observation)

    # Initialize GUI
    gui = PyGameGUI(field_size=8)
    gui.load_environment(env)
    gui.load_agent(agent)

    # Run visualization
    gui.run(num_episodes=3)  # Visualize 3 episodes