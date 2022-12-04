"""
Reporters

These can be added to a neat.Population using `add_reporter()`.
"""
import neat
import numpy as np
import pandas as pd


class StatisticsReporter(neat.StatisticsReporter):

    def __init__(self):
        super(StatisticsReporter, self).__init__()

    def get_scores_per_generation(self):
        """Returns the unaggregated fitnesses across the entire population at each generation."""
        return self.get_fitness_stat(np.array)

    def to_pandas(self):
        generations = self.get_scores_per_generation()
        records = []
        for g, scores in enumerate(generations):
            desc = pd.DataFrame(scores).describe()
            row = {"generation": g + 1,
                   "num_species": len(self.generation_statistics[g]),
                   "best_genome": self.most_fit_genomes[g].key}
            for name, entry in desc.iterrows():
                row[f"fitness.{name}"] = entry.iloc[0]
            records.append(row)
        return pd.DataFrame.from_records(records, index="generation") if records else None
