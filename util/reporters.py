"""
Reporters

These can be added to a neat.Population using `add_reporter()`.
"""

import neat
import numpy as np


class StatisticsReporter(neat.StatisticsReporter):

    def __init__(self):
        super(StatisticsReporter, self).__init__()

    def get_scores_per_generation(self):
        return self.get_fitness_stat(np.array)
