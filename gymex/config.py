"""
Custom Config for Gym Environments
"""
import random

import gym
import neat
import neat.genes
import neat.genome
from neat.config import DefaultClassConfig, ConfigParameter, ConfigParser


class RewardDiscountGenome(neat.DefaultGenome):

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = neat.genes.DefaultNodeGene
        param_dict["connection_gene_type"] = neat.genes.DefaultConnectionGene
        extra_params = [
            ConfigParameter("conn_add_prob_stages", list, []),
            ConfigParameter("conn_delete_prob_stages", list, []),
            ConfigParameter("node_add_prob_stages", list, []),
            ConfigParameter("node_delete_prob_stages", list, []),
            ConfigParameter("generation_stages", list, []),
            ConfigParameter("use_reward_discount", bool, "false"),
        ]
        gconfig = neat.genome.DefaultGenomeConfig(param_dict, extra_params)
        # Parse the list elements.
        first_stage_list = None
        for p in extra_params:
            if p.value_type == list:
                pval = getattr(gconfig, p.name)
                try:
                    pval = [float(v) for v in pval]
                except ValueError as e:
                    raise ValueError(f"Unable to parse {p.name} as float: {pval}\nOriginal error: {e}")
                setattr(gconfig, p.name, pval)
                if p.name.endswith("stages"):
                    if first_stage_list is None:
                        first_stage_list = (p.name, len(pval))
                    else:
                        if len(pval) != first_stage_list[1]:
                            raise ValueError("Found stage lists of different length:\n"
                                             f"- {first_stage_list[0]} (length {first_stage_list[1]})\n"
                                             f"- {p.name} (length {len(pval)})")
        # Do special invariance checking for `generation_stages`.
        if gconfig.generation_stages:
            gconfig.generation_stages = [int(v) for v in gconfig.generation_stages]
            for gen1, gen2 in zip(gconfig.generation_stages, gconfig.generation_stages[1:]):
                if gen1 > gen2:
                    raise ValueError(f"`generation_stages` config param must be a non-decreasing sequence.")
        return gconfig

    def __init__(self, key):
        super().__init__(key)
        self.discount = None

    def configure_new(self, config):
        super().configure_new(config)
        if config.use_reward_discount:
            self.discount = (0.01 + 0.98 * random.random())

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        if config.use_reward_discount:
            self.discount = random.choice((genome1.discount, genome2.discount))

    def mutate(self, config):
        super().mutate(config)
        if config.use_reward_discount:
            self.discount += random.gauss(0.0, 0.05)
            self.discount = max(0.01, min(0.99, self.discount))

    def distance(self, other, config):
        dist = super().distance(other, config)
        if config.use_reward_discount:
            dist += abs(self.discount - other.discount)
        return dist

    def __str__(self):
        disctxt = f"Reward discount: {self.discount}\n" if self.discount is not None else ""
        return disctxt + super().__str__()


class GymConfig(neat.Config):
    """Custom Config for Gym Environments."""

    _SECTION_NAME = "Gym"

    def __init__(self, genome_type, reproduction_type, species_set_type, stagnation_type, filename,
                 config_information=None):
        super().__init__(genome_type, reproduction_type, species_set_type, stagnation_type, filename,
                         config_information)
        self.generation = 0
        self.stage = -1

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

    def update_schedules(self, gennum):
        """
        Updates any configured mutation schedules based on the current generation. Generation is a one-based count.
        """
        self.generation = gennum
        # If we are using our special genome, then apply the mutation rate schedule.
        if self.genome_type == RewardDiscountGenome and self.genome_config.generation_stages:
            # Use generation number to determine what stage we're in.
            # The 0th stage is just the initial values. The first value in the list is "Stage 1".
            current_stage = -1
            for current_stage, sgen in enumerate(self.genome_config.generation_stages):
                if sgen >= self.generation:
                    # This stage comes after the current generation, so the current stage is the previous one.
                    current_stage -= 1
                    break
            if current_stage != self.stage:
                print(f"Advancing mutation schedule from Stage {self.stage + 1} to Stage {current_stage + 1}.")
                self.stage = current_stage
            # Use stage index to determine the desired mutation rates.
            self.genome_config.conn_add_prob = self.genome_config.conn_add_prob_stages[self.stage]
            self.genome_config.conn_delete_prob = self.genome_config.conn_delete_prob_stages[self.stage]
            self.genome_config.node_add_prob = self.genome_config.node_add_prob_stages[self.stage]
            self.genome_config.node_delete_prob = self.genome_config.node_delete_prob_stages[self.stage]

    def save(self, filename):
        super().save(filename)
        # Add one additional section.
        with open(filename, "a") as f:
            f.write(f"\n[{self._SECTION_NAME}]\n")
            self.gym_config.save(f)


def make_config(cfg_path):
    return GymConfig(RewardDiscountGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     cfg_path)
