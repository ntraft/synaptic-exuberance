"""
Viz utils for the OpenAI Lunar Lander.
"""
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

    def display_name(node_id):
        return node_names.get(node_id, str(node_id))

    def label(node_id):
        lbl = display_name(node_id)
        node = genome.nodes.get(node_id)
        if node is not None:
            lbl += f"\n{node.activation}(\n{node.bias:.2f} +\n{node.response:.2f} * {node.aggregation}(inputs))"
        return lbl

    if node_colors is None:
        node_colors = {}
    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2',
    }

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    # Draw nodes.
    inputs = set(list(config.genome_config.input_keys))
    outputs = set(list(config.genome_config.output_keys))
    all_nodes = set(list(config.genome_config.input_keys) + list(genome.nodes.keys()))
    for n in all_nodes:
        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        if n in inputs:
            attrs['fillcolor'] = node_colors.get(n, 'lightgray')
            attrs['shape'] = 'box'
        if n in outputs:
            attrs['fillcolor'] = node_colors.get(n, 'lightblue')
        dot.node(display_name(n), label(n), _attributes=attrs)

    # Draw edges.
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_, output_ = cg.key
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(display_name(input_), display_name(output_), f"{cg.weight:.2f}",
                     _attributes={'style': style, 'color': color, 'penwidth': width})

    # Finally, render and return.
    dot.render(savepath, view=view)
    return dot
