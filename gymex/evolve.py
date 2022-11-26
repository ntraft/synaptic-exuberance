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


def compute_fitness(genome, net, episodes, min_reward, max_reward):
    m = int(round(np.log(0.01) / np.log(genome.discount)))
    discount_function = [genome.discount ** (m - i) for i in range(m + 1)]

    reward_error = []
    for score, data in episodes:
        # Compute normalized discounted reward.
        rewards = np.convolve(data[:, -1], discount_function)[m:]
        rewards = 2 * (rewards - min_reward) / (max_reward - min_reward) - 1.0
        rewards = np.clip(rewards, -1.0, 1.0)

        for row, dr in zip(data, rewards):
            observation = row[:8]
            action = int(row[8])
            output = net.activate(observation)
            reward_error.append(float((output[action] - dr) ** 2))  # squared error from discounted reward value???
            # TODO: This is extremely vulnerable to strange minima. We need to do more than simply predict the reward of
            # TODO: our own policy.

    return reward_error


class PooledErrorCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes = []
        self.generation = 0

        self.min_reward = -200
        self.max_reward = 200

        self.episode_score = []
        self.episode_length = []

    def simulate(self, nets):
        env = gym.make('LunarLander-v2')

        scores = []
        for genome, net in nets:
            observation = env.reset()
            step = 0
            data = []
            while 1:
                step += 1
                if step < 200 and random.random() < 0.2:
                    action = env.action_space.sample()
                else:
                    output = net.activate(observation)
                    action = np.argmax(output)

                observation, reward, done, info = env.step(action)
                data.append(np.hstack((observation, action, reward)))

                if done:
                    break

            data = np.array(data)
            score = np.sum(data[:, -1])
            self.episode_score.append(score)
            scores.append(score)
            self.episode_length.append(step)

            self.test_episodes.append((score, data))

        print(f"Score range [{min(scores):.3f}, {max(scores):.3f}]")

    def evaluate_genomes(self, genomes, config):
        self.generation += 1

        t0 = time.time()
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
                reward_error = compute_fitness(genome, net, self.test_episodes, self.min_reward, self.max_reward)
                genome.fitness = -np.sum(reward_error) / len(self.test_episodes)
        else:
            print(msg + f", asynchronously on {self.num_workers} workers...")
            with multiprocessing.Pool(self.num_workers) as pool:
                jobs = []
                for genome, net in nets:
                    jobs.append(pool.apply_async(compute_fitness,
                                                 (genome, net, self.test_episodes, self.min_reward, self.max_reward)))

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
    ec = PooledErrorCompute(NUM_CORES)
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
                    votes = np.zeros((4,))
                    for n in best_networks:
                        output = n.activate(observation)
                        votes[np.argmax(output)] += 1

                    best_action = np.argmax(votes)
                    observation, reward, done, info = env.step(best_action)
                    score += reward
                    if not os.environ.get("HEADLESS"):
                        env.render()
                    if done:
                        break

                ec.episode_score.append(score)
                ec.episode_length.append(step)

                best_scores.append(score)
                avg_score = np.mean(best_scores)
                print(k, score, avg_score)
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
