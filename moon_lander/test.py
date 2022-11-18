"""
Test the best network and produce a movie of sample runs.
"""

import math
import pickle
import warnings
from pathlib import Path

import neat
import gym.wrappers
import matplotlib.pyplot as plt
import numpy as np

from evolve import make_config, LanderGenome


# TODO: Investigate whether there is some artificial limitation that forces the net to never use the engine??
def make_videos(name, net, result_path, num_videos=5):
    """
    Generate some example videos for the given network.
    """
    env = gym.make('LunarLander-v2')
    # Record one long video for all episodes, but it shouldn't take longer than 60 seconds. Assume 30 fps.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Overwriting existing videos.*")
        env = gym.wrappers.RecordVideo(env, result_path.name, name_prefix=name, video_length=60 * 30)
    try:
        print(f"Generating videos for {name}...")
        outputs = []
        for i in range(1, num_videos + 1):
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
            outputs = np.array(outputs)
            if np.all(np.isclose(outputs, 0.0)):
                print(f"All outputs for {name} were zero!")
            outsz = outputs.shape[1]
            num_cols = int(np.sqrt(outsz))
            num_rows = math.ceil(outsz / num_cols)
            fig, grid = plt.subplots(num_rows, num_cols, squeeze=False, gridspec_kw={"wspace": 0.3, "hspace": 0.5})
            for r, row in enumerate(grid):
                for c, ax in enumerate(row):
                    i = r * num_rows + c
                    vals = outputs[:, i]
                    ax.hist(vals, bins="auto")
                    oid = net.output_nodes[i]
                    ax.set_title(f"Distribution of Output Node {oid}")
                    ax.set_ylabel("Count")
                    ax.set_xlabel(f"Node {oid} Value")
            plt.savefig(result_path / f"{name}-test-outputs.svg")
    finally:
        env.close()


def run():
    local_dir = Path(__file__).parent
    result_path = local_dir / "results"
    config = make_config(local_dir / "config")
    for path in sorted(result_path.glob("winner*.pkl")):
        with open(path, "rb") as f:
            g = pickle.load(f)
            net = neat.nn.FeedForwardNetwork.create(g, config)
            make_videos(path.stem, net, result_path)


if __name__ == '__main__':
    run()
