# Synaptic Exuberance
Biological brains are biased toward growing an excess of neurons/connections in the beginning of life ("synaptic exuberance"), and subsequently pruning away many connections afterward. Can the same idea be helpful to an artificial neural network? Let's find out using the NEAT algorithm for building neural networks.

## Installation
1. Create and activate conda environment.
   - `git clone git@github.com:ntraft/synaptic-exuberance.git`
   - `cd synaptic-exuberance`
   - `conda env create -n <env-name> --file environment.yml`
   - `conda activate <env-name>`
1. Install NEAT from my custom fork (has some required API changes and bug fixes):
   - `cd ..` (Would put it next to `synaptic-exuberance`, but could be anywhere you want.)
   - `git clone git@github.com:ntraft/neat-python.git`
   - `cd neat-python`
   - `pip install -e .`
     - **IMPORTANT:** You must do this in a terminal with the new conda env activated.
1. Test that this runs without crashing (Ctrl-C to quit):
   - `cd ../synaptic-exuberance/gymex`
   - `PYTHONPATH=/path/to/synaptic-exuberance python evolve.py -c config/walker-config`
     - This should create a `gymex/results/` directory with a `generations.pkl` file containing a record of the training process.

## Train a Bipedal Walker
1. Run the `gymex/evolve.py` command using the Walker configuration:
   - `PYTHONPATH=/path/to/synaptic-exuberance python evolve.py -c config/walker-config`
   - (This could take a couple hours, so you may want to set a lower `max_steps`.)
2. When training is finished, it will output the top three "winning" nets as pickle files, along with some plots and diagrams. Typical performance will be around 240-250 points (not very good, but better than random).
3. To test the winners, run `test.py`:
   - `PYTHONPATH=/path/to/synaptic-exuberance python evolve.py -d results`

To run multiple different configs in parallel, create separate results directories, each with a single file called `config`. When you run on these configs, the results will be dropped into the same directory as the corresponding config.
