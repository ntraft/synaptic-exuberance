"""
Custom Config for Gym Environments
"""
import random

import gym
import neat
from neat.config import DefaultClassConfig, ConfigParameter, ConfigParser


class RewardDiscountGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        self.discount = 0.01 + 0.98 * random.random()

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        self.discount += random.gauss(0.0, 0.05)
        self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        disc_diff = abs(self.discount - other.discount)
        return dist + disc_diff

    def __str__(self):
        return f"Reward discount: {self.discount}\n{super().__str__()}"


class GymConfig(neat.Config):
    """Custom Config for Gym Environments."""

    _SECTION_NAME = "Gym"

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename,
                 config_information=None):
        super().__init__(genome_type, reproduction_type, species_set_type, stagnation_type, filename,
                         config_information)

        # NOTE: Unfortunately the superclass doesn't store the parsed params, so we'll just parse again.
        # We can modify the NEAT-Python library if we really want to fix this.
        parameters = ConfigParser()
        with open(filename) as f:
            parameters.read_file(f)

        self.gym_config = DefaultClassConfig(dict(parameters.items(self._SECTION_NAME)),
                                             [ConfigParameter("env_id", str),
                                              ConfigParameter("max_steps", int, 200),
                                              ConfigParameter("num_fitness_episodes", int, 15),
                                              ConfigParameter("new_episode_rate", int, 1),
                                              ConfigParameter("random_action_prob", float, 0.2),
                                              ConfigParameter("fitness_weight", float, 1.0),
                                              ConfigParameter("reward_prediction_weight", float, 0.0),
                                              ConfigParameter("num_best", int, 3),
                                              ConfigParameter("steps_between_eval", int, 5),
                                              ConfigParameter("num_evals", int, 100),
                                              ConfigParameter("eval_ensemble", bool, False),
                                              ConfigParameter("score_threshold", float)])
        # If we are using "reward prediction error" as part of our fitness criterion, ensure the action space is
        # discrete. Reward prediction only makes sense for a discrete action space.
        if self.gym_config.reward_prediction_weight > 0:
            env = gym.make(self.gym_config.env_id)
            if not isinstance(env.action_space, gym.spaces.Discrete):
                raise RuntimeError(f"Reward prediction requires a discrete action space. {self.gym_config.env_id}'s"
                                   " action space is not discrete. To disable reward prediction, set"
                                   " 'reward_prediction_weight' to 0.")
            # We need the reward range to predict normalized rewards.
            # NOTE: Not sure if `reward_threshold` is always valid. We may need to take this as a configured parameter
            # instead.
            self.gym_config.reward_range = [-env.spec.reward_threshold, env.spec.reward_threshold]

    def save(self, filename):
        super().save(filename)
        # Add one additional section.
        with open(filename, "a") as f:
            f.write(f"\n[{self._SECTION_NAME}]\n")
            self.gym_config.save(f)


def make_config(cfg_path):
    return GymConfig(RewardDiscountGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     cfg_path)
