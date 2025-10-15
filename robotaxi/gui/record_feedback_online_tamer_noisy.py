#!/usr/bin/env python3

""" Front-end script for replaying the Snake agent's behavior on a batch of episodes. """

import json
import sys, os
import numpy as np

from robotaxi.gameplay.environment import Environment
from robotaxi.gui.pygame_silent import PyGameGUI
from robotaxi.utils.cli import HelpOnFailArgumentParser
from robotaxi.gameplay.entities import CellType


def parse_command_line_args(args):
    """ Parse command-line arguments and organize them into a single structured object. """

    parser = HelpOnFailArgumentParser(
        description='Snake AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json'
    )

    parser.add_argument(
        '--interface',
        type=str,
        choices=['cli', 'gui'],
        default='gui',
        help='Interface mode (command-line or GUI).',
    )
    parser.add_argument(
        '--agent',
        type=str,
        default='random',
        choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding', 'reward-learning', 'a2c', 'ppo', "tamer-online", "tamer-online-noisy"],
        help='Player agent to use.',
    )
    parser.add_argument(
        '--model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    parser.add_argument(
        '--level',
        type=str,
        default='./robotaxi/levels/8x8-blank.json',
        help='JSON file containing a level definition.',
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1,
        help='The number of episodes to run consecutively.',
    )
    parser.add_argument(
        '--save_frames', 
        action="store_true", 
        default=False, 
        help='save frames as jpg files in screenshots/ folder.'
    )
    parser.add_argument(
        '--stationary', 
        action="store_true", 
        default=False, 
        help='determine whether the environment is stationary'
    )
    parser.add_argument(
        '--collaborating_agent', 
        type=str,
        choices=['human', 'dqn', 'random', 'val-itr', 'mixed', 'one-hot-dqn', 'tile-coding', 'reward-learning', 'a2c'],
        help='Collaborator agent to use.',
    )
    parser.add_argument(
        '--collaborator_model',
        type=str,
        help='File containing a pre-trained agent model.',
    )
    
    parser.add_argument(
        '--participant',
        type=str,
        default='test',
        help='Participant ID.',
    )

    parser.add_argument(
        '--test_run', 
        action="store_true", 
        default=False, 
        help='determine whether the environment is stationary'
    )

    parser.add_argument(
        '--seeds',
        type=str,
        default='42,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60',
        help='Comma-separated list of random seeds for episode objective generation (e.g., "12345,67890,11111"). If empty, random seeds will be generated.'
    )
    
    parser.add_argument(
        '--negative-feedback-only',
        action="store_true",
        default=False,
        help='Only report negative feedbacks to the agent.'
    )
    
    parser.add_argument(
        '--feedback-accuracy',
        type=float,
        default=1.0,
        help='The accuracy of the feedback.'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='The learning rate of the agent.'
    )
    
    parser.add_argument(
        '--lr-decay',
        type=float,
        default=0.998,
        help='The learning rate decay of the agent.'
    )
    
    parser.add_argument(
        '--epsilon-train',
        type=float,
        default=0.2,
        help='The epsilon train of the agent.'
    )
    
    parser.add_argument(
        '--epsilon-test',
        type=float,
        default=0.1,
        help='The epsilon test of the agent.'
    )
    
    parser.add_argument(
        '--weight',
        type=str,
        required=True,
        help='The weight of the agent.'
    )
    
    parser.add_argument(
    "--BCI",
    action="store_true",
    help="Enable BCI mode (receive/send triggers via TiD)"
    )

    parser.add_argument(
        "--calibration",
        action="store_true",
        help="Enable calibration mode"
    )

    parser.add_argument(
        '--threshold',
        type=int,
        default=50,
        help='The probability threshold for user feedback.'
    )
    parsed_args = parser.parse_args(args)
    
    # Convert seeds string to list of integers
    # if parsed_args.seeds start with "trial", the map it to some predefined seeds
    if isinstance(parsed_args.seeds, str) and parsed_args.seeds.startswith("trial"):
        # if it is trial 1, seeds are 1,2,3,4,5,6,7,8,9,10
        # if it is trial 2, seeds are 11,12,13,14,15,16,17,18,19,20
        # ...
        trial_num = int(parsed_args.seeds.split('trial')[1])
        parsed_args.seeds = [(trial_num-1) * 10 + i for i in range(1, 11)]
        # example: trial1 -> 1,2,3,4,5,6,7,8,9,10, trial2 -> 11,12,13,14,15,16,17,18,19,20, ...
    elif isinstance(parsed_args.seeds, str):
        try:
            parsed_args.seeds = [int(seed.strip()) for seed in parsed_args.seeds.split(',') if seed.strip()]
        except ValueError:
            raise ValueError("Seeds must be a comma-separated list of integers or trial1, trial2, trial3, trial4, trial5, trial6")
    else:
        parsed_args.seeds = []
    
    print(f"parsed_args.seeds: {parsed_args.seeds}")
    
    return parsed_args


def create_snake_environment(level_filename, stationary, collaboration, test=False, participant=None):
    """ Create a new Snake environment from the config file. """

    with open(level_filename) as cfg:
        env_config = json.load(cfg)
    if test: env_config["max_step_limit"] = 50

    return Environment(config=env_config, stationary=stationary, collaboration=collaboration, verbose=2, participant=participant)


def load_model(filename):
    """ Load a pre-trained agent model. """

    from keras.models import load_model
    return load_model(filename)


def create_agent(name, model, dimension, env, reward_mapping=None, **kwargs):
    """
    Create a specific type of Snake AI agent.
    
    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model required by certain agents.

    Returns:
        An instance of Snake agent.
    """

    from robotaxi.agent import DeepQNetworkAgent, HumanAgent, RandomActionAgent, ValueIterationAgent, MixedActionAgent, OneHotDQNAgent, TileCodingAgent #, A2CAgent

    if name == 'human':
        return HumanAgent()
    if name == "tamer-online":
        from robotaxi.agent.tamer_agent import OnlineTAMERAgent
        print("Creating TAMER Online Agent")
        return OnlineTAMERAgent()
    if name == "tamer-online-noisy":
        from robotaxi.agent.tamer_agent import OnlineNoisyTAMERAgent
        print("Creating TAMER Online Agent")
        feedback_accuracy = kwargs.get('feedback_accuracy', 0.6)
        negative_feedback_only = kwargs.get('negative_feedback_only', False)
        lr = kwargs.get('lr', 0.01)
        lr_decay = kwargs.get('lr_decay', 0.998)
        epsilon_train = kwargs.get('epsilon_train', 0.2)
        epsilon_test = kwargs.get('epsilon_test', 0.1)
        weight = kwargs.get('weight')
        return OnlineNoisyTAMERAgent(w=weight, feedback_accuracy=feedback_accuracy, negative_feedback_only=negative_feedback_only, alpha=lr, lr_decay=lr_decay, epsilon_train=epsilon_train, epsilon_test=epsilon_test)

        # Example command to run the agent:
        # python record_feedback_online_tamer_noisy.py --agent tamer-online-noisy --model tamer_weights_online_noisy.npy --level 8x8-blank.json --num-episodes 1 --feedback-accuracy 0.6 --negative-feedback-only --lr 0.01 --lr-decay 0.998 --epsilon-train 0.2 --epsilon-test 0.1 --weight w1
    if name == 'dqn':
        if model is None:
            raise ValueError('A model file is required for a DQN agent.')
        return DeepQNetworkAgent(model=model, memory_size=-1, num_last_frames=4)
    if name == 'one-hot-dqn':
        if model is None:
            raise ValueError('A model file is required for an one-hot DQN agent.')
        return OneHotDQNAgent(model=model, memory_size=1000, channels=6)
    if name == 'random':
        return RandomActionAgent()
    if name == 'val-itr':
        return ValueIterationAgent(grid_size=dimension, env=env, reward_mapping=reward_mapping)
    if name == 'mixed':
        return MixedActionAgent(grid_size=dimension, env=env)
    if name == 'tile-coding':
        return TileCodingAgent(weights="tile_coding_weights_660.log")
    if name == 'reward-learning':
        return RewardLearningAgent()
    if name == 'a2c':
        return A2CAgent(grid_size=dimension, env=env)
    if name == 'ppo':
        # from robotaxi.agent import PPOAgent
        from robotaxi.agent.ppo_agent import PPOAgent
        return PPOAgent(model_path="ppo_robottaxi.zip")
    raise KeyError(f'Unknown agent type: "{name}"')


def play_cli(env, agent, agent_name, num_episodes=1):
    """
    Play a set of episodes using the specified Snake agent.
    Use the non-interactive command-line interface and print the summary statistics afterwards.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
    """

    good_fruit_stats = []
    bad_fruit_stats = []
    lava_stats = []
    score_stats = []

    print()
    print('Playing:')

    print('Episode | Score | Good Fruits | Bad Fruits | Lava ')
    for episode in range(num_episodes):
        timestep = env.new_episode()
        agent.begin_episode()
        game_over = False

        while not game_over:
            action = agent.act(timestep.observation, timestep.reward)
            #print(action)
            env.choose_action(action)
            if agent_name == 'mixed':
                timestep = env.timestep(agent_mode=agent.curr_agent)
            else:
                timestep = env.timestep()
            game_over = timestep.is_episode_end

        good_fruit_stats.append(env.stats.good_fruits_eaten)
        bad_fruit_stats.append(env.stats.bad_fruits_eaten)
        lava_stats.append(env.stats.lava_crossed)
        score_stats.append(env.stats.sum_episode_rewards)

        summary = '{:3d}/{:3d} | {:4.1f}  |   {:3d}   |   {:3d}   | {:3d}'
        print(summary.format(episode + 1, num_episodes, env.stats.sum_episode_rewards, env.stats.good_fruits_eaten, env.stats.bad_fruits_eaten, env.stats.lava_crossed))

    print()
    print('Good Fruits eaten {:.1f} +/- {:.1f}'.format(np.mean(good_fruit_stats), np.std(good_fruit_stats)))
    print('Bad Fruits eaten {:.1f} +/- {:.1f}'.format(np.mean(bad_fruit_stats), np.std(bad_fruit_stats)))
    print('Lava eaten {:.1f} +/- {:.1f}'.format(np.mean(lava_stats), np.std(lava_stats)))
    print('Final Score {:.1f} +/- {:.1f}'.format(np.mean(score_stats), np.std(score_stats)))


def play_gui(env, agent, agent_name, num_episodes, save_frames, field_size, collaborating_agent, collaborating_agent_name, participant, test=False, random_seeds=None, calibration = False, BCI = False, threshold=50):
    """
    Play a set of episodes using the specified Snake agent.
    Use the interactive graphical interface.
    
    Args:
        env: an instance of Snake environment.
        agent: an instance of Snake agent.
        num_episodes (int): the number of episodes to run.
        random_seeds (list): list of random seeds for each episode. If None or empty, random seeds will be generated.
    """
    # Convert single random_seed to list format for backward compatibility
    if random_seeds is None:
        random_seeds = []
    gui = PyGameGUI(save_frames=save_frames, field_size=field_size, test=test, random_seeds=random_seeds, calibration = calibration, BCI = BCI, threshold = threshold)
    gui.load_environment(env)
    gui.load_agent(agent, agent_name)
    if collaborating_agent is not None:
        gui.load_collaborator(collaborating_agent, collaborating_agent_name)
    gui.run(num_episodes=num_episodes, participant=participant)
    
    if collaborating_agent is not None:
        print('Final Score {:.1f} '.format(env.stats.sum_episode_rewards+env.stats_collaborator.sum_episode_rewards))

def main():
    parsed_args = parse_command_line_args(sys.argv[1:])

    if not os.path.exists('./csv'): os.makedirs('./csv')
    if not os.path.exists('./log'): os.makedirs('./log')

    collaboration = False if parsed_args.collaborating_agent is None else True
    if collaboration: parsed_args.level = 'robotaxi/levels/8x8-blank-collaboration.json'

    env = create_snake_environment(parsed_args.level, parsed_args.stationary, collaboration, parsed_args.test_run, participant=parsed_args.participant)
    model = load_model(parsed_args.model) if parsed_args.model is not None else None
    dimension = int(parsed_args.level.split('/')[-1].split('x')[0])
    
    if parsed_args.agent == "tamer-online-noisy":
        kwargs = {
            'feedback_accuracy': parsed_args.feedback_accuracy,
            'negative_feedback_only': parsed_args.negative_feedback_only,
            'lr': parsed_args.lr,
            'lr_decay': parsed_args.lr_decay,
            'epsilon_train': parsed_args.epsilon_train,
            'epsilon_test': parsed_args.epsilon_test,
            'weight': parsed_args.weight,
        }
    else:
        kwargs = {}
    agent = create_agent(parsed_args.agent, model, dimension, env, **kwargs)
    print(f"Agent: {agent}")
    collaborator_model = load_model(parsed_args.collaborator_model) if parsed_args.collaborator_model is not None else None
    reward_mapping = {
                CellType.SNAKE_HEAD: 0,
                CellType.SNAKE_BODY: 0,
                CellType.COLLABORATOR_HEAD: 0,
                CellType.COLLABORATOR_BODY: 0,
                CellType.GOOD_FRUIT: -5,
                CellType.BAD_FRUIT: -1,
                CellType.LAVA: 6,
                CellType.EMPTY: 0,
                CellType.PIT: 0,
                CellType.WALL: -100,
            }
    collaborating_agent = create_agent(parsed_args.collaborating_agent, collaborator_model, dimension, env, reward_mapping=reward_mapping) if collaboration else None
    print(f"Collaborating Agent: {parsed_args.collaborating_agent}")
    print(f"parsed_args.interface: {parsed_args.interface}")
    if parsed_args.interface == 'cli':
        play_cli(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes)
    else:
        play_gui(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes, save_frames=parsed_args.save_frames, field_size=dimension, collaborating_agent=collaborating_agent, collaborating_agent_name=parsed_args.collaborating_agent, participant=parsed_args.participant, test=parsed_args.test_run, random_seeds=parsed_args.seeds, calibration = parsed_args.calibration, BCI = parsed_args.BCI, threshold = parsed_args.threshold)

if __name__ == '__main__':
    main()
