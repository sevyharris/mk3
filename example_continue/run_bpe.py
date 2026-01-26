# example script to run Bayesian Parameter Estimation
# This one lets the user continue where they left off
import os
import numpy as np
import ess


# define the observed data and associated uncertainties D, P(D)
observed_data_x = np.array([600, 1100, 1400])
observed_data_y = np.array([360500, 580500, 1620500])
observed_data_y_uncertainties = np.array([200000, 300000, 200000])

# define the priors and associated uncertainties H, P(H)
priors = np.array([200, 500])
prior_uncertainties = np.array([100, 200])


# define the simulation function
def simulationFunction(x, a, b):
    x = np.array(x)
    y = (x-a) ** 2 + b
    return y


def sim_wrapper(parameters):
    a_given = parameters[0]
    b_given = parameters[1]
    y = simulationFunction(observed_data_x, a_given, b_given)  #an alternatie simpler syntax to unpack the parameters would be: simulationFunction(x_values_for_data, *parametersArray)
    return y


optimizer = ess.BPEstimator(
    sim_wrapper,
    priors,
    prior_uncertainties,
    observed_data_x,
    observed_data_y,
    observed_data_y_uncertainties,
    results_dir=os.path.join(os.path.dirname(__file__), 'results'),
    parameter_names=['a', 'b'],
    plot_dir=os.path.join(os.path.dirname(__file__), 'plots'),
    load_save_point=True
)

N_samples = 1000
optimizer.collect_samples(N_samples)

# # only execute this at the end (don't do these two if you're going to continue and run more)
optimizer.compile_and_flatten_chains()
optimizer.make_all_plots()
