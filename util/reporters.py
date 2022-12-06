"""
Reporters

These can be added to a neat.Population using `add_reporter()`.
"""
from collections import namedtuple

import neat
import numpy as np
import pandas as pd

ChangeType = namedtuple("ChangeType", ("fitness_direction", "node_direction", "conn_direction"))


def fraction_of(condition, iterable):
    iterable = list(iterable)
    if len(iterable) == 0:
        return np.nan
    return len([x for x in iterable if condition(x)]) / len(iterable)


def record_success_rate(action, condition, changes, record):
    target_changes = list(filter(condition, changes))
    record[f"{action}_success_rate"] = fraction_of(lambda c: c.fitness_direction > 0, target_changes)
    record[f"{action}_failure_rate"] = fraction_of(lambda c: c.fitness_direction < 0, target_changes)


class StatisticsReporter(neat.StatisticsReporter):

    def __init__(self, neat_algo):
        """
        Args:
            neat_algo (neat.Population): The Population object which will be issuing events to this reporter.
        """
        super().__init__()
        self.neat_algo = neat_algo
        self.prev_population = {}
        self.generation_records = []

    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)

        record = {
            "generation": len(self.generation_statistics),  # 1-based count
            "num_species": len(species.species),
            "pop_size": len(population),
            "best_genome": best_genome.key,
            "conn_add_prob": config.genome_config.conn_add_prob,
            "conn_delete_prob": config.genome_config.conn_delete_prob,
            "node_add_prob": config.genome_config.node_add_prob,
            "node_delete_prob": config.genome_config.node_delete_prob,
        }

        scores = []
        conn_sizes = []
        node_sizes = []
        changes = []
        for genome in population.values():
            # Record the fitness.
            scores.append(genome.fitness)

            # Info about the size of the network.
            pruned = genome.get_pruned_copy(config.genome_config)
            conn_sizes.append(len(pruned.connections))
            node_sizes.append(len(pruned.nodes))
            if genome == best_genome:
                record["best_genome_num_conns"] = len(pruned.connections)
                record["best_genome_num_nodes"] = len(pruned.nodes)

            # Some info about how this child differs from its parents.
            parent_ids = self.neat_algo.reproduction.ancestors[genome.key]
            if self.prev_population and parent_ids:
                p1, p2 = self.prev_population.get(parent_ids[0]), self.prev_population.get(parent_ids[1])
                if p1 is not None and p2 is not None:
                    # Ensure p1 is the fitter parent. (NOTE: Important for this to be `>=` and not `>`. It has to match
                    # what is done in `DefaultGenome.configure_crossover()`.)
                    if p2.fitness >= p1.fitness:
                        p1, p2 = p2, p1
                    if genome.fitness > p1.fitness:
                        fitness_direction = 1
                    elif genome.fitness < p2.fitness:
                        fitness_direction = -1
                    else:
                        # Being in between the fitnesses of the two parents is considered "no change".
                        fitness_direction = 0
                    # NOTE: We know the size of children before mutation is the same as their fittest parent (p1).
                    node_direction = len(genome.nodes) - len(p1.nodes)
                    conn_direction = len(genome.connections) - len(p1.connections)
                    changes.append(ChangeType(fitness_direction, node_direction, conn_direction))

        # Store all summary stats for these measures across the whole population.
        for metric, values in (("fitness", scores), ("node", node_sizes), ("conn", conn_sizes)):
            desc = pd.DataFrame(values).describe()
            for stat, entry in desc.iterrows():
                if stat != "count":
                    record[f"{metric}.{stat}"] = entry.iloc[0]

        # Store extra summaries over just the most fit individuals.
        top5 = sorted(population.values(), key=lambda g: g.fitness)[:5]
        top5_conns = [len(g.connections) for g in top5]
        top5_nodes = [len(g.nodes) for g in top5]
        for name, sizes in (("node", top5_nodes), ("conn", top5_conns)):
            sizes = np.array(sizes)
            record[name + ".top5_mean"] = np.mean(sizes)
            record[name + ".top5_std"] = np.std(sizes)
            record[name + ".top5_min"] = np.min(sizes)
            record[name + ".top5_max"] = np.max(sizes)

        # Store success rates of structural mutations.
        record_success_rate("node_add", lambda c: c.node_direction > 0, changes, record)
        record_success_rate("node_remove", lambda c: c.node_direction < 0, changes, record)
        record_success_rate("conn_add", lambda c: c.conn_direction > 0, changes, record)
        record_success_rate("conn_remove", lambda c: c.conn_direction < 0, changes, record)
        record_success_rate("weight_only", lambda c: c.node_direction == 0 and c.conn_direction == 0, changes, record)

        self.generation_records.append(record)
        self.prev_population = population

    def get_scores_per_generation(self):
        """Returns the unaggregated fitnesses across the entire population at each generation."""
        return self.get_fitness_stat(np.array)

    def to_pandas(self):
        if not self.generation_records:
            return None
        return pd.DataFrame.from_records(self.generation_records, index="generation")
