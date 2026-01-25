# Let's sketch out generally how this will go

# imports
import os
import time
import numpy as np
import yaml
import zeus
import mpi4py
import scipy.stats
import plotting

use_dill = True
if use_dill:  # Use this if you get the `TypeError: cannot pickle 'module' object` error
    import dill
    global MPI
    
    from mpi4py import MPI as _MPI

    _MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
    MPI = _MPI

# probably have to handle MPI stuff here

RANK = MPI.COMM_WORLD.Get_rank()
np.random.seed(400 + RANK)


observed_data_x_values = np.array([600,1100,1400])
observed_data_y_values = np.array([360500, 580500, 1620500])
observed_data_y_values_uncertainties = np.array([200000, 300000, 200000])
if observed_data_y_values_uncertainties.ndim == 1:
    observed_data_y_values_uncertainties = np.diag(np.float_power(observed_data_y_values_uncertainties, 2.0))


priors = np.array([200, 500])
prior_uncertainties = np.array([100, 200])

# convert to covariance matrix if it's a list of std devs
if prior_uncertainties.ndim == 1:
    prior_uncertainties = np.diag(np.float_power(prior_uncertainties, 2.0))


# load settings
# for now, these are constants just saved here, but later we'll let the user set them elsewhere
N_ZEUS_CHAINS = 2  # zeus paper recommends nchains=2 or 4
N_WALKERS = 8
N_SAMPLES = 1000
N_ENSEMBLE_STEPS = int(N_SAMPLES / N_WALKERS)
N_PARAMETERS = len(priors)



def simulationFunction(x,a,b): #here x is a scalar or an array and "a" and "b" are constants for the equation.
    #time.sleep(0.05)
    x =np.array(x)
    y = (x-a)**2 + b
    return y


def sim_wrapper(parameters):
    a_given = parameters[0]
    b_given = parameters[1]
    y = simulationFunction(observed_data_x_values, a_given, b_given)  #an alternatie simpler syntax to unpack the parameters would be: simulationFunction(x_values_for_data, *parametersArray)
    return y


def get_log_prior(parameters):
    # return the log of the prior
    # returns log P(H)
    log_prior = scipy.stats.multivariate_normal.logpdf(
        x=parameters,
        mean=priors,
        cov=prior_uncertainties
    )
    return log_prior


def get_log_likelihood(sim_results):
    # returns log P(D|H)
    log_likelihood = scipy.stats.multivariate_normal.logpdf(
        x=sim_results,
        mean=observed_data_y_values,
        cov=observed_data_y_values_uncertainties
    )
    return log_likelihood


def get_log_posterior(parameters):
    # has to return log probability (can be unnormalized) of posterior
    # this is log probability of hypothesis + log probability of data given hypothesis
    # can ignore P(D) in Bayes theorem because this is constant
    # log P(H) + log(D|H)
    log_prior = get_log_prior(parameters)
    sim_results = sim_wrapper(parameters)
    log_likelihood = get_log_likelihood(sim_results)
    log_posterior = log_prior + log_likelihood
    # print(log_posterior)
    return log_posterior


if mpi4py.MPI.COMM_WORLD.Get_size() < 2:
    N_ZEUS_CHAINS = 1
    # raise ValueError('You gotta use MPI for now. At least 2 processes')

walker_start_points = np.random.multivariate_normal(priors, prior_uncertainties, size=N_WALKERS, check_valid='warn', tol=1e-8)  # probably want uncertainties to be lower than usual
# print(RANK, walker_start_points)

if RANK == 0:
    print("number of chains: ", N_ZEUS_CHAINS)
    print("MCMC walkers: ", N_WALKERS)
    print("numParameters: ", N_PARAMETERS)
    print("walker start points shape: ", walker_start_points.shape)
    print("nEnsembleSteps", N_ENSEMBLE_STEPS)


# set up the zeus sampler
with zeus.ChainManager(N_ZEUS_CHAINS) as cm:
    zeus_sampler = zeus.EnsembleSampler(
        N_WALKERS, N_PARAMETERS, logprob_fn=get_log_posterior, 
        pool=cm.get_pool, maxiter=1e6  # maxiter should be 1e6 instead of 1e4 ashi says
    )

    # run the zeus sampler
    zeus_sampler.run_mcmc(walker_start_points, N_ENSEMBLE_STEPS)

    zeus_sampler.summary

    # save files
    chain = zeus_sampler.get_chain(flat=True, discard=0.1)
    logPs = zeus_sampler.get_log_prob(flat=True, discard=0.1)

    # samples = zeus_sampler.samples.flatten(discard=0.1)
    np.save(f'chain_{RANK}.npy', chain)
    np.save(f'logPs_{RANK}.npy', logPs)

    
# print(samples.shape)

# do analysis
chain0 = np.load('chain_0.npy')
print(chain0.shape)
logP0 = np.load('logPs_0.npy')
print(logP0.shape)

MAP_index = np.argmin(np.abs(logP0 - np.max(logP0)))
print('MAP:', logP0[MAP_index], chain0[MAP_index, :])

outdir = '/home/moon/mk3/'

# maximalist plotting
# plotting.make_histograms(
#     outdir,
#     chain0,
#     MAP=chain0[MAP_index, :],
#     mean=np.nanmean(chain0, axis=0),
#     initial=priors,
#     parameter_names=['a', 'b']
# )

# minimal hist
plotting.make_histograms(outdir, chain0)