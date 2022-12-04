"""
Reporters

These can be added to a neat.Population using `add_reporter()`.
"""
import neat
import numpy as np
import pandas as pd


class StatisticsReporter(neat.StatisticsReporter):

    def __init__(self):
        super().__init__()
        self.generation_records = []

    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)

        record = {
            "generation": len(self.generation_statistics),  # 1-based count
            "num_species": len(species.species),
            "best_genome": best_genome.key,
            "conn_add_prob": config.genome_config.conn_add_prob,
            "conn_delete_prob": config.genome_config.conn_delete_prob,
            "node_add_prob": config.genome_config.node_add_prob,
            "node_delete_prob": config.genome_config.node_delete_prob,
        }

        scores = []
        conn_sizes = []
        node_sizes = []
        for genome in population.values():
            scores.append(genome.fitness)
            genome = genome.get_pruned_copy(config.genome_config)
            conn_sizes.append(len(genome.connections))
            node_sizes.append(len(genome.nodes))
        # Store all fitness summary stats.
        desc = pd.DataFrame(scores).describe()
        for name, entry in desc.iterrows():
            record[f"fitness.{name}"] = entry.iloc[0]
        # Store just a few summary stats about network size.
        for name, sizes in (("node", node_sizes), ("conn", conn_sizes)):
            sizes = np.array(sizes)
            record[name + ".mean"] = sizes.mean()
            record[name + ".std"] = sizes.std()
            # noinspection PyArgumentList
            record[name + ".min"] = sizes.min()
            # noinspection PyArgumentList
            record[name + ".max"] = sizes.max()

        self.generation_records.append(record)

    def get_scores_per_generation(self):
        """Returns the unaggregated fitnesses across the entire population at each generation."""
        return self.get_fitness_stat(np.array)

    def to_pandas(self):
        if not self.generation_records:
            return None
        return pd.DataFrame.from_records(self.generation_records, index="generation")
