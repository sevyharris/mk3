# Let's sketch out generally how this will go

# imports
import os
import numpy as np
import dill

# import mpi4py
import scipy.stats
import plotting
import zeus
import zeus.parallel

from mpi4py import MPI

# global MPI
# from mpi4py import MPI as _MPI
# _MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
# MPI = _MPI

# probably have to handle MPI stuff here
N_PROCESSORS = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
np.random.seed(400 + RANK)


class BPEstimator():
    """
    Bayesian Parameter Estimator using zeus ensemble slice sampling.

    Parameters
    ----------
    simulation_fn : callable
        A function that takes parameters and returns simulated observations.
        Should accept a 1D numpy array of parameters and return predictions
        at the observed data points
    priors : array_like
        Mean values of the prior distribution for each parameter
    prior_uncertainties : array_like
        Prior uncertainties for each parameter. Can be either:
        - 1D array of standard deviations (converted to diagonal covariance matrix)
        - 2D covariance matrix
    observed_data_x : array_like
        Independent variable values where observations were made
    observed_data_y : array_like
        Observed data values to fit
    observed_data_y_uncertainties : array_like
        Uncertainties in observed data. Can be either:
        - 1D array of standard deviations (converted to diagonal covariance matrix)
        - 2D covariance matrix
    results_dir : str, optional
        Directory path for saving MCMC results. Defaults to current directory
    parameter_names : list of str, optional
        Names of the parameters for plotting purposes

    Notes
    -----
    The estimator assumes Gaussian distributions for both priors and likelihoods.
    MCMC samples are saved as 'chain_{i}.npy' and 'logPs_{i}.npy' files.
    """
    def __init__(
        self,
        simulation_fn,
        priors,
        prior_uncertainties,
        observed_data_x,
        observed_data_y,
        observed_data_y_uncertainties,
        results_dir=None,
        parameter_names=None,
        plot_dir=None,
    ):
        self.simulation_fn = simulation_fn
        self.priors = priors
        self.prior_uncertainties = prior_uncertainties
        self.observed_data_x = observed_data_x
        self.observed_data_y = observed_data_y
        self.observed_data_y_uncertainties = observed_data_y_uncertainties
        self.parameter_names = parameter_names
        # convert 1D array of uncertainties (standard deviations) into 2D covariance matrix
        if self.prior_uncertainties.ndim == 1:
            self.prior_uncertainties = np.diag(np.float_power(self.prior_uncertainties, 2.0))
        if self.observed_data_y_uncertainties.ndim == 1:
            self.observed_data_y_uncertainties = np.diag(np.float_power(self.observed_data_y_uncertainties, 2.0))

        # load settings
        # for now, these are constants just saved here, but later we'll let the user set them elsewhere
        self.N_ZEUS_CHAINS = 2  # zeus paper recommends nchains=2 or 4
        self.N_PARAMETERS = len(self.priors)

        # allow users to run without MPI
        if N_PROCESSORS < 2:
            self.N_ZEUS_CHAINS = 1

        self.results_dir = './results'
        if results_dir is not None:
            self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

        self.plot_dir = './plots'
        if plot_dir is not None:
            self.plot_dir = plot_dir
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)


    # EXPANSION - could make some sort of factory for using other kinds of distributions
    def get_log_prior(self, parameters):
        # return the log of the prior
        # returns log P(H)
        log_prior = scipy.stats.multivariate_normal.logpdf(
            x=parameters,
            mean=self.priors,
            cov=self.prior_uncertainties
        )
        return log_prior

    def get_log_likelihood(self, sim_results):
        # returns log P(D|H)
        log_likelihood = scipy.stats.multivariate_normal.logpdf(
            x=sim_results,
            mean=self.observed_data_y,
            cov=self.observed_data_y_uncertainties
        )
        return log_likelihood


    def get_log_posterior(self, parameters):
        # has to return log probability (can be unnormalized) of posterior
        # this is log probability of hypothesis + log probability of data given hypothesis
        # can ignore P(D) in Bayes theorem because this is constant
        # log P(H) + log(D|H)
        log_prior = self.get_log_prior(parameters)
        sim_results = self.simulation_fn(parameters)
        log_likelihood = self.get_log_likelihood(sim_results)
        log_posterior = log_prior + log_likelihood
        return log_posterior

    def get_processor_to_chain_index(self, N_proc, N_chains):
        # returns a dictionary converting processor name to zeus chain index
        # collect the processor ranges dividing the different chains
        ranges = []
        for i, ranks in zeus.parallel.split_ranks(N_proc, N_chains):
            ranges.append(ranks[0])

        proc2chain = {}
        for i, primary_proc in enumerate(ranges):
            proc2chain[primary_proc] = i
        return proc2chain


    def collect_samples(self, N_samples, N_walkers=8, discard=0.1):
        N_ENSEMBLE_STEPS = int(N_samples / N_walkers)

        # EXTENSION other ways to initialize? uniform distribution?
        walker_start_points = np.random.multivariate_normal(
            self.priors,
            self.prior_uncertainties,
            size=N_walkers,
            check_valid='warn',
            tol=1e-8
        )

        if RANK == 0:
            print("number of chains: ", self.N_ZEUS_CHAINS)
            print("MCMC walkers: ", N_walkers)
            print("numParameters: ", self.N_PARAMETERS)
            print("walker start points shape: ", walker_start_points.shape)
            print("nEnsembleSteps", N_samples)

        # set up the zeus sampler
        with zeus.ChainManager(self.N_ZEUS_CHAINS) as cm:
            zeus_sampler = zeus.EnsembleSampler(
                N_walkers,
                self.N_PARAMETERS,
                logprob_fn=self.get_log_posterior, 
                pool=cm.get_pool,
                maxiter=1e6
            )

            # run the zeus sampler
            zeus_sampler.run_mcmc(walker_start_points, N_ENSEMBLE_STEPS)

            # maybe turn this off, or implement better logging options
            zeus_sampler.summary

            # save files
            chain = zeus_sampler.get_chain(flat=False, discard=discard)
            logPs = zeus_sampler.get_log_prob(flat=False, discard=discard)

            # convert from processor index to chain index
            proc2chain = self.get_processor_to_chain_index(N_PROCESSORS, self.N_ZEUS_CHAINS)
            if RANK in proc2chain.keys():
                chain_index = proc2chain[RANK]
                np.save(os.path.join(self.results_dir, f'chain_{chain_index}.npy'), chain)
                np.save(os.path.join(self.results_dir, f'logPs_{chain_index}.npy'), logPs)

    def make_all_plots(self):
        if RANK > 0:
            return

        # load just the first chain for now
        chain0 = np.load(os.path.join(self.results_dir, 'chain_0.npy'))
        logP0 = np.load(os.path.join(self.results_dir, 'logPs_0.npy'))

        # https://github.com/minaskar/zeus/blob/1abdf08252a99e9aa186dcee414f559624b3bafd/zeus/samples.py#L90
        # Copied from zeus sample.flatten()
        flattened_chain0 = chain0.reshape((-1, 2), order='F')
        flattened_logP0 = logP0.reshape((-1,), order='F')

        # find MAP
        MAP_index = np.argmin(np.abs(flattened_logP0 - np.max(flattened_logP0)))
        print('MAP:', flattened_logP0[MAP_index], flattened_chain0[MAP_index, :])

        # maximalist plotting
        plotting.make_histograms(
            self.plot_dir,
            flattened_chain0,
            MAP=flattened_chain0[MAP_index, :],
            mean=np.nanmean(flattened_chain0, axis=0),
            initial=self.priors,
            parameter_names=self.parameter_names
        )

        # # corner plot / posterior scatter matrix
        # plotting.make_corner_plot(plot_dir, flattened_chain0)

        # corner plot / posterior scatter matrix
        plotting.make_posterior_scatter_matrix(self.plot_dir, flattened_chain0)

        # heat scatter matrix
        plotting.make_heat_scatter(
            self.plot_dir,
            flattened_chain0,
            MAP=flattened_chain0[MAP_index, :],
            mean=np.nanmean(flattened_chain0, axis=0),
            initial=self.priors,
            parameter_names=self.parameter_names
        )

        # maximal autocorrelation - needs full chain shape
        plotting.make_autocorrelation(self.plot_dir, chain0, parameter_names=self.parameter_names)