"""
Test the performance of the best genome produced by evolve-ctrnn.py.
"""

import pickle
from pathlib import Path

import neat
from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie

from evolve_ctrnn import make_config


local_dir = Path(__file__).parent
result_path = local_dir / "results"

# load the winner
with open(result_path / "winner-ctrnn.pkl", "rb") as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file.
config = make_config(local_dir / "config-ctrnn")

# Instantiate the simulation.
sim = CartPole()
net = neat.ctrnn.CTRNN.create(c, config, sim.time_step)

print()
print("Initial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

# Run the given simulation for up to 120 seconds.
balance_time = 0.0
while sim.t < 120.0:
    inputs = sim.get_scaled_state()
    action = net.advance(inputs, sim.time_step, sim.time_step)

    # Apply action to the simulated cart-pole
    force = discrete_actuator_force(action)
    sim.step(force)

    # Stop if the network fails to keep the cart within the position or angle limits.
    # The per-run fitness is the number of time steps the network can balance the pole
    # without exceeding these limits.
    if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
        break

    balance_time = sim.t

print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

print()
print("Final conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()
print("Making movie...")
make_movie(net, discrete_actuator_force, 15.0, result_path / "ctrnn-movie.mp4")
