#!/usr/bin/env python3

""" Front-end script for replaying the Snake agent's behavior on a batch of episodes. """

import json
import sys, os
import numpy as np

from robotaxi.gameplay.environment import Environment
from robotaxi.gui.pygame_silent import PyGameGUI
from robotaxi.utils.cli import HelpOnFailArgumentParser
from robotaxi.gameplay.entities import CellType


import yaml
import argparse
import utils


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


def create_agent(name, tamer_kwargs=None):
    """
    Create a specific type of Snake AI agent.
    
    Args:
        name (str): key identifying the agent type.
        model: (optional) a pre-trained model required by certain agents.

    Returns:
        An instance of Snake agent.
    """

    from robotaxi.agent import DeepQNetworkAgent, HumanAgent, RandomActionAgent, ValueIterationAgent, MixedActionAgent, OneHotDQNAgent, TileCodingAgent #, A2CAgent

    if name == "tamer-online-noisy":
        from robotaxi.agent.tamer_agent import OnlineNoisyTAMERAgent
        print("Creating TAMER Online Agent")
        return OnlineNoisyTAMERAgent(**tamer_kwargs)
        # Example command to run the agent:
        # python record_feedback_online_tamer_noisy.py --agent tamer-online-noisy --model tamer_weights_online_noisy.npy --level 8x8-blank.json --num-episodes 1 --feedback-accuracy 0.6 --negative-feedback-only --lr 0.01 --lr-decay 0.998 --epsilon-train 0.2 --epsilon-test 0.1 --weight w1
    
    if name == 'human':
        return HumanAgent()
    
    if name == "tamer-online":
        raise ValueError("this TAMER Online Agent is not supported anymore")
        # from robotaxi.agent.tamer_agent import OnlineTAMERAgent
        # print("Creating TAMER Online Agent")
        # return OnlineTAMERAgent()
        
    if name == 'random':
        return RandomActionAgent()
    
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


def play_gui(env, agent, agent_name, num_episodes, save_frames, field_size, collaborating_agent, collaborating_agent_name, participant, test=False, random_seeds=None, calibration = False, BCI = False, threshold = 50):
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

def build_args_parser():
    """
    Schema-only parser that matches config.yaml.
    No defaults here; YAML provides them.
    Unset CLI args are suppressed from the Namespace.
    """
    parser = argparse.ArgumentParser(
        description='Snake AI replay client.',
        epilog='Example: play.py --agent dqn --model dqn-final.model --level 10x10.json',
        argument_default=argparse.SUPPRESS,  # 🔑 suppress attributes not passed on CLI
    )

    # -------- Interface & run control --------
    parser.add_argument('--interface', type=str, choices=['cli', 'gui'],
                        help='Interface mode (command-line or GUI).')
    parser.add_argument('--agent', type=str,
                        choices=['human', 'dqn', 'random', 'val-itr', 'mixed',
                                 'one-hot-dqn', 'tile-coding', 'reward-learning',
                                 'a2c', 'ppo', 'tamer-online', 'tamer-online-noisy'],
                        help='Player agent to use.')
    parser.add_argument('--model', type=str,
                        help='Path to a pre-trained agent model.')
    parser.add_argument('--level', type=str,
                        help='JSON file containing a level definition.')
    parser.add_argument('--num-episodes', type=int,
                        help='Number of episodes to run consecutively.')
    parser.add_argument('--save_frames', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Save frames as JPG files in screenshots/ folder.')
    parser.add_argument('--stationary', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Whether the environment is stationary.')
    parser.add_argument('--participant', type=str,
                        help='Participant ID.')
    parser.add_argument('--test_run', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Whether this is a test-only run.')
    parser.add_argument('--seeds', type=str,
                        help='Comma-separated random seeds string.')

    # -------- Collaborator settings --------
    parser.add_argument('--collaborating_agent', type=str,
                        choices=['human', 'dqn', 'random', 'val-itr', 'mixed',
                                 'one-hot-dqn', 'tile-coding', 'reward-learning', 'a2c'],
                        help='Collaborator agent to use.')
    parser.add_argument('--collaborator_model', type=str,
                        help='Path to a pre-trained collaborator model.')

    # -------- Feedback / human-in-the-loop / BCI --------
    parser.add_argument('--negative-feedback-only', '--negative_feedback_only',
                        dest='negative_feedback_only',
                        action='store_true', default=argparse.SUPPRESS,
                        help='Only report negative feedback to the agent.')
    parser.add_argument('--feedback-accuracy', type=float,
                        help='Global feedback accuracy (if used).')
    parser.add_argument('--feedback-tpr', type=float,
                        help='True positive rate for feedback.')
    parser.add_argument('--feedback-tnr', type=float,
                        help='True negative rate for feedback.')
    parser.add_argument('--threshold', type=float,
                        help='Probability threshold (0–100) for user feedback.')
    parser.add_argument('--BCI', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Enable BCI mode (send/receive triggers via TiD).')
    parser.add_argument('--calibration', action='store_true',
                        default=argparse.SUPPRESS,
                        help='Enable calibration mode.')
    parser.add_argument('--feedback-processor', type=str,
                        help="Feedback preprocessor class name (e.g., 'TINYMLFeedbackPreProcessor').")
    parser.add_argument('--margin', type=float,
                        help="Margin for feedback preprocessor.")

    # -------- Learning hyperparameters --------
    parser.add_argument('--lr', type=float, help='Learning rate.')
    parser.add_argument('--lr-decay', type=float, help='Learning rate decay.')
    parser.add_argument('--epsilon-train', type=float, help='Epsilon during training.')
    parser.add_argument('--epsilon-test', type=float, help='Epsilon during evaluation.')
    parser.add_argument('--uncertainty-bonus-scale', type=float,
                        help='Scale for uncertainty exploration bonus.')

    # -------- Weights / features / modes --------
    parser.add_argument('--weight', type=str,
                        help="Path or identifier to agent's weight.")
    parser.add_argument('--feature_version', type=str,
                        help='Feature set version (e.g., v2, v4).')
    parser.add_argument('--mode', type=str,
                        help="Run mode: e.g., 'train', 'eval', 'replay'.")

    # -------- Paths --------
    parser.add_argument('--save_path', type=str,
                        help='Where to save logs/checkpoints/outputs.')

    return parser


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    
    # 1) Parse only control flags first: --config and --overwrite
    ctrl = argparse.ArgumentParser(add_help=True)
    ctrl.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ctrl.add_argument("--overwrite", action="store_true",
                      help="Allow CLI to overwrite conflicting YAML values")
    ctrl_args, remaining_args = ctrl.parse_known_args()
    
    with open(ctrl_args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}
    
    cli_only_parser = build_args_parser()
    cli_ns, extra = cli_only_parser.parse_known_args(remaining_args)
    if extra:
        # If you expect no extras, you can error out here
        ctrl.error(f"Unrecognized arguments: {' '.join(extra)}")
        
    cli_args = vars(cli_ns)           # only explicitly provided CLI keys
    allowed_keys = {a.dest for a in build_args_parser()._actions if a.dest != "help"}

    # 4) Warn on unknown YAML keys
    unknown_yaml = set(yaml_cfg) - allowed_keys
    if unknown_yaml:
        print(f"[WARN] Unknown YAML keys ignored: {sorted(unknown_yaml)}", file=sys.stderr)

    # 5) Start from YAML (primary), then merge CLI with conflict detection
    final_cfg = {k: v for k, v in yaml_cfg.items() if k in allowed_keys}
    conflicts = {}

    for k, v_cli in cli_args.items():
        if k in final_cfg and final_cfg[k] != v_cli:
            if not ctrl_args.overwrite:
                conflicts[k] = (final_cfg[k], v_cli)
            else:
                final_cfg[k] = v_cli
        else:
            # Not present in YAML, or same value → just set it
            final_cfg[k] = v_cli

    if conflicts:
        lines = ["Conflict detected between YAML and CLI (use --overwrite to allow):"]
        for k, (yval, cval) in conflicts.items():
            lines.append(f"  {k}: YAML={yval!r}  CLI={cval!r}")
        ctrl.error("\n".join(lines))

    # 6) At this point final_cfg is resolved under your policy
    print("Final configuration:")
    for k in sorted(final_cfg):
        print(f"{k}: {final_cfg[k]}")
    

    if not os.path.exists('./csv'): os.makedirs('./csv')
    if not os.path.exists('./log'): os.makedirs('./log')
    
    final_cfg['seeds'] = [int(seed.strip()) for seed in final_cfg['seeds'].split(',') if seed.strip()]
    parsed_args = utils.to_namespace(final_cfg)
    
    collaboration = False if parsed_args.collaborating_agent is None else True
    env = create_snake_environment(parsed_args.level, parsed_args.stationary, collaboration, parsed_args.test_run, participant=parsed_args.participant)
    model = load_model(parsed_args.model) if parsed_args.model is not None else None
    dimension = int(parsed_args.level.split('/')[-1].split('x')[0])
    
    if parsed_args.agent == "tamer-online-noisy":
        tamer_kwargs = final_cfg
    else:
        tamer_kwargs = None
    agent = create_agent(parsed_args.agent, tamer_kwargs)
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
    
    # collaborating_agent = create_agent(parsed_args.collaborating_agent, collaborator_model, dimension, env, reward_mapping=reward_mapping) if collaboration else None
    collaborating_agent = None
    print(f"Collaborating Agent: {parsed_args.collaborating_agent}")
    print(f"parsed_args.interface: {parsed_args.interface}")
    if parsed_args.interface == 'cli':
        play_cli(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes)
    else:
        # print(f"parsed_args.BCI: {parsed_args.BCI}")
        # import time; time.sleep(10)
        play_gui(env, agent, parsed_args.agent, num_episodes=parsed_args.num_episodes, save_frames=parsed_args.save_frames, field_size=dimension, collaborating_agent=collaborating_agent, collaborating_agent_name=parsed_args.collaborating_agent, participant=parsed_args.participant, test=parsed_args.test_run, random_seeds=parsed_args.seeds, calibration = parsed_args.calibration, BCI = parsed_args.BCI, threshold = parsed_args.threshold)

if __name__ == '__main__':
    main()
