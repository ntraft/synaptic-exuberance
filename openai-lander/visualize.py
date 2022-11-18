import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_fitness(stats, ylog=False, view=False, savepath="fitness.svg"):
    """ Plots the population's average and best fitness. """
    generation = range(len(stats.most_fit_genomes))
    plt.plot(generation, stats.get_fitness_stat(min), 'b:', label="min")
    plt.plot(generation, stats.get_fitness_stat(max), 'b:', label="max")

    avg_fitness = np.array(stats.get_fitness_mean())
    plt.plot(generation, avg_fitness, 'b-', label="average")

    # stdev_fitness = np.array(stats.get_fitness_stdev())
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    # plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")

    best_fitness = [c.fitness for c in stats.most_fit_genomes]
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    if savepath:
        plt.savefig(savepath)
    if view:
        plt.show()

    plt.close()


def plot_species(stats, view=False, savepath="speciation.svg"):
    """ Visualizes speciation throughout evolution. """
    species_sizes = stats.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    if savepath:
        plt.savefig(savepath)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, savepath=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # If requested, use a copy of the genome which omits all components that won't affect the output.
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}
    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}
    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input_, output_ = cg.key
            a = node_names.get(input_, str(input_))
            b = node_names.get(output_, str(output_))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    if savepath:
        dot.render(savepath, view=view)

    return dot
