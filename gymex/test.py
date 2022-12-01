"""
Test the best network and produce a movie of sample runs.
"""
import os
import pickle
import sys
import warnings
from pathlib import Path

import neat
import gym.wrappers
import numpy as np

import util.argparsing as argutils
import gymex.visualize as visualize
from gymex.config import make_config, RewardDiscountGenome
from gymex.evolve import take_step


def make_videos(name, net, gym_config, result_path, num_episodes=5):
    """
    Generate some example videos for the given network.
    """
    env = gym.make(gym_config.env_id)
    if not os.environ.get("HEADLESS"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Overwriting existing videos.*")
            # Record one long video for all episodes, but it shouldn't take longer than 60 seconds. Assume 30 fps.
            env = gym.wrappers.RecordVideo(env, result_path.name, name_prefix=name, video_length=60 * 30)
    try:
        print(f"Generating videos for {name}...")
        outputs = []
        rewards = []
        for i in range(1, num_episodes + 1):
            step = 0
            episode_rewards = []
            observation = env.reset()
            done = False
            while not done:
                step += 1
                output, _, observation, reward, done, _ = take_step(env, observation, net)
                outputs.append(output)
                episode_rewards.append(reward)
            rewards.append(episode_rewards)
            print(f"    Score for episode {i}: {sum(episode_rewards)}")

        # Plot stats.
        if rewards:
            visualize.plot_reward_trajectories(rewards, savepath=result_path / f"{name}-rewards.svg")
        if outputs:
            visualize.plot_net_outputs(outputs, net, savepath=result_path / f"{name}-test-outputs.svg")
    finally:
        env.close()


def main(argv=None):
    parser = argutils.create_parser(__doc__)
    parser.add_argument("-d", "--results-dir", metavar="PATH", type=argutils.existing_dir,
                        help="Directory where results are stored. (default: same as config)")
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, default="./config",
                        help="NEAT config file.")
    parser.add_argument("-m", "--model", metavar="FILENAME", default="winner*.pkl",
                        help="The model(s) to test. This should be a filename relative to the --results-dir. You may"
                             " also supply a glob pattern to match multiple models. The string 'random' is a special"
                             " value, indicating we should test a randomly instantiated NEAT genome.")
    parser.add_argument("-n", "--num-episodes", metavar="INT", type=int, default=5,
                        help="Number of episodes to run on each model.")
    parser.add_argument("-g", "--write-graph", action="store_true", help="Also save a visualization of the network(s).")
    args = parser.parse_args(argv)
    user_supplied_args = parser.get_user_specified_args()

    # If results dir is user-specified but config is not, find the config in that directory.
    if args.results_dir and args.results_dir.is_dir() and "config" not in user_supplied_args:
        args.config = args.results_dir / "config"
    # Otherwise, use the existing config value. If results dir was not specified, use the config dir.
    if not args.results_dir:
        args.results_dir = args.config.parent
    
    config = make_config(args.config)
    if args.model.startswith("random"):
        maybe_num = args.model[6:]  # remove "random"
        num_models = int(maybe_num) if maybe_num else 1  # e.g. "random6" = six random models
        args.model = "random-*.pkl"
        for i in range(num_models):
            g = RewardDiscountGenome(0)
            g.configure_new(config.genome_config)
            with open(args.results_dir / f"random-{i}.pkl", "wb") as f:
                pickle.dump(g, f)
    for path in sorted(args.results_dir.glob(args.model)):
        with open(path, "rb") as f:
            g = pickle.load(f)
            visualize.draw_net(config, g, savepath=args.results_dir / f"{path.stem}-net-pruned.gv", prune_unused=True)
            net = neat.nn.FeedForwardNetwork.create(g, config)
            make_videos(path.stem, net, config.gym_config, args.results_dir, args.num_episodes)


if __name__ == '__main__':
    sys.exit(main())
