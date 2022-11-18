"""
Test the best network and produce a movie of sample runs.
"""

import pickle
import warnings
from pathlib import Path

import neat
import gym.wrappers
import numpy as np

from evolve import make_config, LanderGenome


# TODO: Possibly output actual action values for debugging?
# TODO: Investigate whether there is some artificial limitation that forces the net to never use the engine??
def make_videos(name, net, num_videos=5):
    """
    Generate some example videos for the given network.
    """
    env = gym.make('LunarLander-v2')
    # Record one long video for all episodes, but it shouldn't take longer than 60 seconds. Assume 30 fps.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*Overwriting existing videos.*")
        env = gym.wrappers.RecordVideo(env, "results", name_prefix=name, video_length=60 * 30)
    try:
        print(f"Generating videos for {name}...")
        for i in range(1, num_videos + 1):
            step = 0
            score = 0
            observation = env.reset()
            done = False
            while not done:
                step += 1
                output = net.activate(observation)
                action = np.argmax(output)
                observation, reward, done, info = env.step(action)
                score += reward
            print(f"    Score for episode {i}: {score}")
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
            make_videos(path.stem, net)


if __name__ == '__main__':
    run()
