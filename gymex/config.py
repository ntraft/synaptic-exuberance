"""
Custom Config for Gym Environments
"""
import random

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
                                             [ConfigParameter('env_id', str),
                                              ConfigParameter('num_best', int, 3),
                                              ConfigParameter('steps_between_eval', int, 5),
                                              ConfigParameter('num_evals', int, 100),
                                              ConfigParameter('score_threshold', float),
                                              ConfigParameter('random_action_prob', float, 0.2),
                                              ConfigParameter('reward_range', list, None)])
        if self.gym_config.reward_range:
            self.gym_config.reward_range[0] = float(self.gym_config.reward_range[0])
            self.gym_config.reward_range[1] = float(self.gym_config.reward_range[1])

    def save(self, filename):
        super().save(filename)
        # Add one additional section.
        with open(filename, 'a') as f:
            f.write(f'\n[{self._SECTION_NAME}]\n')
            self.gym_config.save(f)


def make_config(cfg_path):
    return GymConfig(RewardDiscountGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     cfg_path)
