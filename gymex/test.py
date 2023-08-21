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

import util.argparsing as argutils
import gymex.visualize as visualize
from gymex.config import make_config, RewardDiscountGenome
from gymex.evolve import take_step


def make_videos(name, net, gym_config, result_path, num_episodes=5, max_seconds=30):
    """
    Generate some example videos for the given network.
    """
    env = gym.make(gym_config.env_id, render_mode="rgb_array")
    if not os.environ.get("HEADLESS"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Overwriting existing videos.*")
            # Record one long video for all episodes, but it shouldn't take longer than N seconds. Assume 30 fps.
            env = gym.wrappers.RecordVideo(env, str(result_path), name_prefix=name, video_length=max_seconds * 30)
    try:
        print(f"Generating videos for {name}...")
        outputs = []
        rewards = []
        for i in range(1, num_episodes + 1):
            step = 0
            episode_rewards = []
            observation, _ = env.reset()
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
    argutils.add_experiment_args(parser, True)
    parser.add_argument("-m", "--model", metavar="FILENAME", default="winner*.pkl",
                        help="The model(s) to test. This should be a filename relative to the --results-dir. You may"
                             " also supply a glob pattern to match multiple models. The string 'random' is a special"
                             " value, indicating we should test a randomly instantiated NEAT genome.")
    parser.add_argument("-n", "--num-episodes", metavar="INT", type=int, default=5,
                        help="Number of episodes to run on each model.")
    parser.add_argument("-s", "--seconds", metavar="NUM", type=float, default=30,
                        help="Maximum length of the video for each model, in seconds.")
    parser.add_argument("-g", "--write-graph", action="store_true", help="Also save a visualization of the network(s).")
    args = parser.parse_args(argv)
    argutils.resolve_experiment_args(parser, args, __file__)
    
    config = make_config(args.config)
    print(f"Saving results to {args.results_dir}.")
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
            make_videos(path.stem, net, config.gym_config, args.results_dir, args.num_episodes, args.seconds)


if __name__ == '__main__':
    sys.exit(main())
