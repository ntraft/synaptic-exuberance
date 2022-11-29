"""
Evolve a control/reward estimation network for the Moon Lander example from OpenAI Gym.
LunarLander-v2 environment (https://gym.openai.com/envs/LunarLander-v2).
Sample run here: https://gym.openai.com/evaluations/eval_FbKq5MxAS9GlvB7W6ioJkg
"""
import multiprocessing
import os
import pickle
import random
import sys
import time
from collections import namedtuple
from pathlib import Path

import neat
import gym.wrappers
import matplotlib.pyplot as plt
import numpy as np

import util.argparsing as argutils
import util.reporters as reporters
import gymex.visualize as visualize
from gymex.config import make_config


NUM_CORES = os.cpu_count()
if hasattr(os, "sched_getaffinity"):
    # This function is only available on certain platforms. When running with Slurm, it can tell us the true number of
    # cores we have access to.
    NUM_CORES = len(os.sched_getaffinity(0))


def take_step(env, observation, networks, random_action_prob=0.0):
    if not isinstance(networks, (tuple, list)):
        networks = [networks]
    assert len(networks) > 0 and networks[0] is not None

    # Run the networks.
    num_outputs = len(networks[0].output_nodes)
    all_actions = np.zeros((len(networks), num_outputs))
    for i, n in enumerate(networks):
        all_actions[i] = n.activate(observation)

    # Choose an action.
    if random.random() < random_action_prob:
        # Take a random action with some probability.
        action = env.action_space.sample()
    else:
        if isinstance(env.action_space, gym.spaces.Discrete):
            # Make a choice from a discrete set of actions. If multiple networks, take the action with the highest
            # total value.
            # TODO: Maybe vote instead of summing reward estimates?
            action = np.argmax(all_actions.sum(axis=0))
        elif isinstance(env.action_space, gym.spaces.Box):
            # Output a continuous action. If multiple networks, average them.
            action = all_actions.mean(axis=0)
        else:
            raise RuntimeError(f"Unsupported action space: {env.action_space}")

    observation, reward, done, info = env.step(action)
    return all_actions, action, observation, reward, done, info


def compute_reward_prediction_error(args):
    genome, net, episodes, reward_range = args

    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for ep in episodes:
        # Compute normalized discounted reward.
        rewards = np.convolve(ep.rewards, discount_function)[m:]
        if reward_range:
            # NOTE: This is done knowing that we are usually using networks whose outputs are clamped to [-1, 1].
            # So we need targets in the same range.
            min_reward, max_reward = reward_range
            rewards = 2 * (rewards - min_reward) / (max_reward - min_reward) - 1.0
            rewards = np.clip(rewards, -1.0, 1.0)

        for obs, act, dr in zip(ep.observations, ep.actions, rewards):
            output = net.activate(obs)
            reward_error.append(float((output[act] - dr) ** 2))  # squared error from discounted reward value???
            # TODO: This is extremely vulnerable to strange minima. We need to do more than simply predict the reward of
            # TODO: our own policy.

    return np.mean(reward_error)


Episode = namedtuple("Episode", ["observations", "actions", "rewards"])


def run_sim_episodes(args):
    net, gym_config = args
    env = gym.make(gym_config.env_id)
    episodes = []
    for _ in range(gym_config.num_fitness_episodes):
        observation = env.reset()
        observations = []
        actions = []
        rewards = []
        while True:
            _, action, observation, reward, done, info = take_step(env, observation, net,
                                                                   gym_config.random_action_prob)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:
                break

        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        episodes.append(Episode(observations, actions, rewards))

    return episodes


class PooledErrorCompute(object):

    def __init__(self, gym_config, pool=None):
        self.gym_config = gym_config
        self.pool = pool
        self.generation = 0
        self.test_episodes = []

    def simulate(self, nets, gym_config):
        msg = f"Simulating {len(nets)} nets in {gym_config.num_fitness_episodes} episodes"
        jobs = [(net, gym_config) for _, net in nets]
        if self.pool:
            print(msg + f", asynchronously...")
            results = self.pool.map(run_sim_episodes, jobs)
        else:
            print(msg + ", serially...")
            results = map(run_sim_episodes, jobs)

        for i, episodes in enumerate(results):
            genome = nets[i][0]
            scores = [ep.rewards.sum() for ep in episodes]
            genome.fitness = sum(scores) / gym_config.num_fitness_episodes
            self.test_episodes.extend(episodes)

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        nets = [(g, neat.nn.FeedForwardNetwork.create(g, config)) for gid, g in genomes]
        print(f"Time to create {len(nets)} networks: {time.time() - t0:.2f}s")
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison. However, if we are using the score as part of the
        # fitness computation (`fitness_weight > 0`), then we always need to run this part.
        if config.gym_config.fitness_weight > 0 or self.generation % config.gym_config.new_episode_rate == 1:
            # Keep the episodes from the past two generations.
            num_to_keep = 2 * len(nets) * config.gym_config.num_fitness_episodes
            self.test_episodes = self.test_episodes[-num_to_keep:]
            self.simulate(nets, config.gym_config)
            print(f"Simulation run time: {time.time() - t0:.2f}s")
            t0 = time.time()

        if config.gym_config.reward_prediction_weight > 0.0:
            # Assign a composite fitness to each genome; genomes can make progress either
            # by improving their total reward or by making more accurate reward estimates.
            msg = f"Evaluating {len(nets)} nets on {len(self.test_episodes)} test episodes"
            jobs = [(genome, net, self.test_episodes, config.gym_config.reward_range) for genome, net in nets]
            if self.pool:
                print(msg + f", asynchronously...")
                results = self.pool.map(compute_reward_prediction_error, jobs)
            else:
                print(msg + ", serially...")
                results = map(compute_reward_prediction_error, jobs)
            for i, pred_error in enumerate(results):
                genome = nets[i][0]
                if config.gym_config.fitness_weight > 0:
                    genome.fitness = (config.gym_config.fitness_weight * genome.fitness -
                                      config.gym_config.reward_prediction_weight * pred_error)
                else:
                    genome.fitness = -config.gym_config.reward_prediction_weight * pred_error

            print(f"Reward prediction compute time: {time.time() - t0:.2f}s\n")


def run_evolution(config, result_dir):
    """
    Run until the winner from a generation is able to solve the environment or the user interrupts the process.
    """
    env = None
    pool = None
    best_genomes = None

    try:
        print(f'Creating Gym environment "{config.gym_config.env_id}".')
        env = gym.make(config.gym_config.env_id)
        if NUM_CORES > 1:
            print(f"Spawning a pool of {NUM_CORES} processes.")
            pool = multiprocessing.Pool(NUM_CORES)
        else:
            pool = None
        ec = PooledErrorCompute(config.gym_config, pool)

        pop = neat.Population(config)
        stats = reporters.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.StdOutReporter(True))
        # Checkpoint every 25 generations or 900 seconds.
        pop.add_reporter(neat.Checkpointer(25, 900))

        steps_between_eval = config.gym_config.steps_between_eval

        while True:
            gen_best = pop.run(ec.evaluate_genomes, steps_between_eval)

            # print(gen_best)

            visualize.plot_fitness(stats, savepath=result_dir / "fitness.svg")
            visualize.plot_species(stats, savepath=result_dir / "speciation.svg")

            mfs = np.mean(stats.get_fitness_mean()[-steps_between_eval:])
            print(f"Average mean fitness over last {steps_between_eval} generations: {mfs}")
            mfs = np.mean(stats.get_fitness_stat(min)[-steps_between_eval:])
            print(f"Average min fitness over last {steps_between_eval} generations: {mfs}")

            # Evaluate the best genomes seen so far.
            best_genomes = stats.best_unique_genomes(config.gym_config.num_best)
            best_networks = [neat.nn.FeedForwardNetwork.create(g, config) for g in best_genomes]

            # If we want to evaluate as an ensemble, just make this a list of one list, so all networks are used
            # together in one ensemble.
            if config.gym_config.eval_ensemble:
                best_networks = [best_networks]

            solved = [True] * len(best_networks)
            for i, net in enumerate(best_networks):
                ensemble_text = f" (ensemble of {len(net)} networks)" if isinstance(net, list) else ""
                print(f"Testing network {i}" + ensemble_text + "...")
                scores = []
                for k in range(config.gym_config.num_evals):
                    observation = env.reset()
                    score = 0
                    step = 0
                    done = False
                    while not done:
                        step += 1
                        # Use the total reward estimates from all the best networks to
                        # determine the best action given the current state.
                        _, _, observation, reward, done, _ = take_step(env, observation, net)
                        score += reward
                        if not os.environ.get("HEADLESS"):
                            env.render()

                    scores.append(score)
                    avg_score = np.mean(scores)
                    print(f"    Test Episode {k}: score = {score}, avg so far = {avg_score}")
                    if score < config.gym_config.score_threshold:
                        # We must always get above the threshold, else stop early and decide we aren't solved yet.
                        solved[i] = False
                        break

            if np.any(solved):
                msg = "Solved by: "
                for i, s in enumerate(solved):
                    if s:
                        msg += f"winner-{best_networks[i]}, "
                print(msg[:-2])
                return best_genomes

        # end while

    except KeyboardInterrupt:
        print("User requested termination.")
        return best_genomes

    finally:
        if env:
            env.close()
        if pool:
            pool.terminate()
            pool.join()


def main(argv=None):
    parser = argutils.create_parser(__doc__)
    local_dir = Path(__file__).parent
    parser.add_argument("-d", "--results-dir", metavar="PATH", type=Path, default=local_dir / "results",
                        help="Directory where results are stored.")
    parser.add_argument("-c", "--config", metavar="PATH", type=argutils.existing_path, default=local_dir / "config",
                        help="NEAT config file.")
    args = parser.parse_args(argv)

    config = make_config(args.config)
    result_path = args.results_dir.resolve()
    result_path.mkdir(exist_ok=True)

    best_genomes = run_evolution(config, result_path)

    # Save the winners.
    if best_genomes:
        print(f"Saving the {len(best_genomes)} best genomes.")
        for n, g in enumerate(best_genomes):
            name = f"winner-{n}"
            with open(result_path / f"{name}.pkl", "wb") as f:
                pickle.dump(g, f)

            visualize.draw_net(config, g, view=False, savepath=result_path / f"{name}-net.gv")
            visualize.draw_net(config, g, view=False, savepath=result_path / f"{name}-net-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    sys.exit(main())
