# example script to run Bayesian Parameter Estimation
# This one lets the user continue where they left off
import os
import numpy as np
import pandas as pd
import ess
import sys
# sys.path.append('/projects/westgroup/harris.se/uncertainty_estimator/bpe/simulation/')
sys.path.append('/home/moon/uncertainty_estimator/bpe/simulation/')  # Cantera simulation
import sim_wrapper
import yaml


working_dir = os.path.dirname(os.path.abspath(__file__))

# Load all the experiment + uncertainty csvs
experiment_df = pd.read_csv(os.path.join(working_dir, 'experiment.csv'))
observed_data_y = experiment_df.values

experiment_uncertainty_df = pd.read_csv(os.path.join(working_dir, 'experiment_uncertainty.csv'))
observed_data_y_uncertainties = experiment_uncertainty_df.values

MIN_STD = 1e-5
observed_data_y_uncertainties[observed_data_y_uncertainties < MIN_STD] = MIN_STD  # avoid small uncertainties

# Load all the prior + uncertainty csvs
priors_df = pd.read_csv(os.path.join(working_dir, 'priors.csv'))
priors = priors_df.values

prior_uncertainty_df = pd.read_csv(os.path.join(working_dir, 'priors_uncertainty.csv'))
prior_uncertainties = prior_uncertainty_df.values

# Load other info for simulation
sim_info_yaml = os.path.join(working_dir, 'sim_info.yaml')
with open(sim_info_yaml) as f:
    sim_info = yaml.safe_load(f)
parameter_names = sim_info['parameter_names']
out_gas_indices = sim_info['out_gas_indices']
observed_data_x = sim_info['sample_distances']


optimizer = ess.BPEstimator(
    sim_wrapper.simulation_wrapper,
    priors,
    prior_uncertainties,
    observed_data_x,
    observed_data_y,
    observed_data_y_uncertainties,
    results_dir=os.path.join(os.path.dirname(__file__), 'results'),
    parameter_names=parameter_names,
    plot_dir=os.path.join(os.path.dirname(__file__), 'plots'),
    load_save_point=False
)

print('y shape', optimizer.observed_data_y.shape)
print('prior shape', optimizer.priors.shape)
print('y uncertainties shape', optimizer.observed_data_y_uncertainties.shape)
print('prior uncertainties shape', optimizer.prior_uncertainties.shape)

# print('y data', optimizer.observed_data_y)
# print('y uncertainties', optimizer.observed_data_y_uncertainties)
# print('prior uncertainties', optimizer.prior_uncertainties)

# try a walker initialization
walker_start_points = np.random.multivariate_normal(
    optimizer.priors,
    optimizer.prior_uncertainties,
    size=1,
    check_valid='warn',
    tol=1e-8
)

# print('walker start points', walker_start_points)
# logP = optimizer.get_log_posterior(walker_start_points[0])
# print('log P=', logP)



N_samples = 25
optimizer.collect_samples(N_samples)

# # only execute this at the end (don't do these two if you're going to continue and run more)
optimizer.compile_and_flatten_chains()
optimizer.make_all_plots()
