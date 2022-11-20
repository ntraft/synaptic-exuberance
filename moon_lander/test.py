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
import visualize
from evolve import make_config, LanderGenome


def make_videos(name, net, result_path, num_episodes=5):
    """
    Generate some example videos for the given network.
    """
    env = gym.make('LunarLander-v2')
    if not os.environ.get("HEADLESS"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*Overwriting existing videos.*")
            # Record one long video for all episodes, but it shouldn't take longer than 60 seconds. Assume 30 fps.
            env = gym.wrappers.RecordVideo(env, result_path.name, name_prefix=name, video_length=60 * 30)
    try:
        print(f"Generating videos for {name}...")
        outputs = []
        for i in range(1, num_episodes + 1):
            step = 0
            score = 0
            observation = env.reset()
            done = False
            while not done:
                step += 1
                output = net.activate(observation)
                outputs.append(output)
                action = np.argmax(output)
                observation, reward, done, info = env.step(action)
                score += reward
            print(f"    Score for episode {i}: {score}")

        # Plot histogram of each output.
        if outputs:
            visualize.plot_net_outputs(outputs, name, net, result_path)
    finally:
        env.close()


def main(argv=None):
    parser = argutils.create_parser(__doc__)
    local_dir = Path(__file__).parent
    parser.add_argument("-d", "--results-dir", metavar="PATH", type=argutils.existing_dir,
                        default=local_dir / "results", help="Directory where results are stored.")
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, default=local_dir / "config",
                        help="NEAT config file.")
    parser.add_argument("-m", "--model", metavar="FILENAME", default="winner*.pkl",
                        help="The model(s) to test. This should be a filename relative to the --results-dir. You may"
                             " also supply a glob pattern to match multiple models. The string 'random' is a special"
                             " value, indicating we should test a randomly instantiated NEAT genome.")
    parser.add_argument("-n", "--num-episodes", metavar="INT", type=int, default=5,
                        help="Number of episodes to run on each model.")
    parser.add_argument("-g", "--write-graph", action="store_true", help="Also save a visualization of the network(s).")
    args = parser.parse_args(argv)
    config = make_config(args.config)
    if args.model == "random":
        args.model = "random-net.pkl"
        g = LanderGenome(0)
        g.configure_new(config.genome_config)
        with open(args.results_dir / args.model, "wb") as f:
            pickle.dump(g, f)
    for path in sorted(args.results_dir.glob(args.model)):
        with open(path, "rb") as f:
            g = pickle.load(f)
            visualize.draw_net(config, g, savepath=args.results_dir / f"{path.stem}-net-pruned.gv", prune_unused=True)
            net = neat.nn.FeedForwardNetwork.create(g, config)
            make_videos(path.stem, net, args.results_dir, args.num_episodes)


if __name__ == '__main__':
    sys.exit(main())