"""
Single-pole balancing experiment using a continuous-time recurrent neural network (CTRNN).
"""

import multiprocessing
import pickle
from pathlib import Path

import cart_pole
import neat
import visualize

runs_per_net = 5
simulation_seconds = 60.0
time_const = cart_pole.CartPole.time_step


# Use the CTRNN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)

    fitnesses = []
    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()
        net.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.advance(inputs, time_const, time_const)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)

        # print(f"{net} fitness {fitness}")

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def make_config(config_path):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       config_path)


def run():
    # Load the config file, which is assumed to live in the same directory as this script.
    local_dir = Path(__file__).parent
    config = make_config(local_dir / "config-ctrnn")

    result_path = local_dir / "results"
    result_path.mkdir(exist_ok=True)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open(result_path / "winner-ctrnn.pkl", "wb") as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=True, savepath=result_path / "ctrnn-fitness.svg")
    visualize.plot_species(stats, view=True, savepath=result_path / "ctrnn-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       savepath=result_path / "winner-ctrnn.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       savepath=result_path / "winner-ctrnn-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()