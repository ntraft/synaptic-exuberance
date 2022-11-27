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


def compute_fitness(genome, net, episodes, reward_range):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, (observations, actions, rewards) in episodes:
        # Compute normalized discounted reward.
        rewards = np.convolve(rewards, discount_function)[m:]
        if reward_range:
            # NOTE: This is done knowing that we are usually using networks whose outputs are clamped to [-1, 1].
            # So we need targets in the same range.
            min_reward, max_reward = reward_range
            rewards = 2 * (rewards - min_reward) / (max_reward - min_reward) - 1.0
            rewards = np.clip(rewards, -1.0, 1.0)

        for obs, act, dr in zip(observations, actions, rewards):
            output = net.activate(obs)
            reward_error.append(float((output[act] - dr) ** 2))  # squared error from discounted reward value???
            # TODO: This is extremely vulnerable to strange minima. We need to do more than simply predict the reward of
            # TODO: our own policy.

    return reward_error


class PooledErrorCompute(object):
    def __init__(self, gym_config, num_workers):
        self.gym_config = gym_config
        self.num_workers = num_workers
        self.test_episodes = []
        self.reward_range = None
        self.generation = 0

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        env = gym.make(self.gym_config.env_id)
        # TODO: Weird thing we're doing for historical compatibility but probably want to remove later.
        self.reward_range = [-env.spec.reward_threshold, env.spec.reward_threshold]

        scores = []
        for genome, net in nets:
            observation = env.reset()
            step = 0
            observations = []
            actions = []
            rewards = []
            while 1:
                step += 1
                _, action, observation, reward, done, info = take_step(env, observation, net,
                                                                       self.gym_config.random_action_prob)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)

                if done:
                    break

            observations = np.array(observations)
            actions = np.array(actions)
            rewards = np.array(rewards)
            score = rewards.sum()
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, (observations, actions, rewards)))

        print(f"Score range [{min(scores):.3f}, {max(scores):.3f}]")

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
        # TODO: Paralellize this and make the compute_fitness optional (only applies to discrete action spaces).
        nets = [(g, neat.nn.FeedForwardNetwork.create(g, config)) for gid, g in genomes]
        print(f"Time to create {len(nets)} networks: {time.time() - t0:.2f}s")
        t0 = time.time()

        # Periodically generate a new set of episodes for comparison.
        if self.generation % 10 == 1:
            # Keep at most 300 episodes.
            self.test_episodes = self.test_episodes[-300:]
            self.simulate(nets)
            print(f"Simulation run time: {time.time() - t0:.2f}s")
            t0 = time.time()

        # Assign a composite fitness to each genome; genomes can make progress either
        # by improving their total reward or by making more accurate reward estimates.
        msg = f"Evaluating {len(nets)} nets on {len(self.test_episodes)} test episodes"
        if self.num_workers < 2:
            print(msg + ", serially...")
            for genome, net in nets:
                reward_error = compute_fitness(genome, net, self.test_episodes, self.reward_range)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            print(msg + f", asynchronously on {self.num_workers} workers...")
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(compute_fitness,
                                                 (genome, net, self.test_episodes, self.reward_range)))

                for job, (genome_id, genome) in zip(jobs, genomes):
                    reward_error = job.get(timeout=None)
                    genome.fitness = -np.sum(reward_error) / len(self.test_episodes)

        print(f"Final fitness compute time: {time.time() - t0:.2f}s\n")


def run_evolution(config, result_dir):
    """
    Run until the winner from a generation is able to solve the environment or the user interrupts the process.
    """
    env = gym.make(config.gym_config.env_id)
    pop = neat.Population(config)
    stats = reporters.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # Checkpoint every 25 generations or 900 seconds.
    pop.add_reporter(neat.Checkpointer(25, 900))
    ec = PooledErrorCompute(config.gym_config, NUM_CORES)
    steps_between_eval = config.gym_config.steps_between_eval
    best_genomes = None

    while 1:
        try:
            gen_best = pop.run(ec.evaluate_genomes, steps_between_eval)

            # print(gen_best)

            visualize.plot_fitness(stats, savepath=result_dir / "fitness.svg")
            visualize.plot_species(stats, savepath=result_dir / "speciation.svg")

            plt.plot(ec.episode_score, 'g-', label='score')
            plt.plot(ec.episode_length, 'b-', label='length')
            plt.grid()
            plt.legend(loc='best')
            plt.savefig(result_dir / "scores.svg")
            plt.close()

            mfs = np.mean(stats.get_fitness_mean()[-steps_between_eval:])
            print(f"Average mean fitness over last {steps_between_eval} generations: {mfs}")
            mfs = np.mean(stats.get_fitness_stat(min)[-steps_between_eval:])
            print(f"Average min fitness over last {steps_between_eval} generations: {mfs}")

            # Use the best genomes seen so far as an ensemble-ish control system.
            # TODO: Weird? Cheating? Could conflict with each other? Doesn't properly represent the real result?
            best_genomes = stats.best_unique_genomes(config.gym_config.num_best)
            best_networks = [neat.nn.FeedForwardNetwork.create(g, config) for g in best_genomes]

            solved = True
            best_scores = []
            for k in range(config.gym_config.num_evals):
                observation = env.reset()
                score = 0
                step = 0
                while 1:
                    step += 1
                    # Use the total reward estimates from all the best networks to
                    # determine the best action given the current state.
                    _, _, observation, reward, done, _ = take_step(env, observation, best_networks)
                    score += reward
                    if not os.environ.get("HEADLESS"):
                        env.render()
                    if done:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = np.mean(best_scores)
                print(f"Test Episode {k}: score = {score}, avg so far = {avg_score}")
                if avg_score < config.gym_config.score_threshold:
                    # As soon as our average score drops below this threshold, stop early and decide we aren't good
                    # enough yet.
                    # TODO: This is a super weird criterion. This would mean a different ordering of scores could be
                    # judged differently.
                    solved = False
                    break

            if solved:
                print("Solved.")
                return best_genomes

        except KeyboardInterrupt:
            print("User requested termination.")
            return best_genomes

        finally:
            env.close()


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
        print(f"Saving {len(best_genomes)} best genomes.")
        for n, g in enumerate(best_genomes):
            name = f"winner-{n}"
            with open(result_path / f"{name}.pkl", "wb") as f:
                pickle.dump(g, f)

            visualize.draw_net(config, g, view=False, savepath=result_path / f"{name}-net.gv")
            visualize.draw_net(config, g, view=False, savepath=result_path / f"{name}-net-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    sys.exit(main())
