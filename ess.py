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
        Directory path for saving sample chain and logP .npy files
    parameter_names : list of str, optional
        Names of the parameters for plotting purposes
    plot_dir : str, optional
        Directory path for saving plots
    load_save_point : bool, False
        Start sampling where last run left off?

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
        load_save_point=False,
    ):
        self.simulation_fn = simulation_fn
        self.priors = np.array(priors)
        self.prior_uncertainties = np.array(prior_uncertainties)
        self.observed_data_x = np.array(observed_data_x)
        self.observed_data_y = np.array(observed_data_y)
        self.observed_data_y_uncertainties = np.array(observed_data_y_uncertainties)
        self.parameter_names = parameter_names

        # if priors are multidimensional, flatten them
        if self.priors.ndim > 1:
            self.priors = self.priors.ravel()
        # if prior uncertainties are multidimensional and not square, flatten them
        if self.prior_uncertainties.ndim > 1 and \
                self.prior_uncertainties.shape[0] != self.prior_uncertainties.shape[1]:
            self.prior_uncertainties = self.prior_uncertainties.ravel()

        # if you have multidimensional ys, then stack it into a single array
        if self.observed_data_y.ndim > 1:
            self.observed_data_y = self.observed_data_y.ravel()
        # assume that a non-square matrix was intended to be a list of multidimensional outputs
        if self.observed_data_y_uncertainties.ndim > 1 and \
                self.observed_data_y_uncertainties.shape[0] != self.observed_data_y_uncertainties.shape[1]:
            self.observed_data_y_uncertainties = self.observed_data_y_uncertainties.ravel()

        # convert 1D array of uncertainties (standard deviations) into 2D covariance matrix
        if self.prior_uncertainties.ndim == 1:
            self.prior_uncertainties = np.diag(np.float_power(self.prior_uncertainties, 2.0))
        if self.observed_data_y_uncertainties.ndim == 1:
            self.observed_data_y_uncertainties = np.diag(np.float_power(self.observed_data_y_uncertainties, 2.0))

        # np.save('ys.npy', self.observed_data_y)
        # np.save('y_uncertainties.npy', self.observed_data_y_uncertainties)


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

        self.load_save_point = load_save_point
        # if we're loading a previous savepoint, need to check the shape of things

        # check for existing chains if loading save point, but don't actually load here
        if self.load_save_point and RANK == 0:
            print("Loading previous save point...")

            # load one of the existing chains to check shape
            chain0_file = os.path.join(self.results_dir, 'chain_0.npy')
            if not os.path.exists(chain0_file):
                print('Previous chain file does not exist, so we start from the beginning')
                self.load_save_point = False
                # raise FileNotFoundError(f"Cannot find existing chain file {chain0_file} for loading save point.")
            else:
                chain0 = np.load(chain0_file)

                # check shape
                if chain0.shape[2] != self.N_PARAMETERS:
                    raise ValueError(
                        f"Loaded chain parameter dimension {chain0.shape} does not match current prior dimension {self.N_PARAMETERS}."
                    )
                # probably not the end of the world if N_walkers or N_samples differ
                print("Save point loaded successfully.")


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

    def get_log_likelihood(self, sim_results, independent=True):
        # if you have multiple ys, you should stack them
        # returns log P(D|H)
        #print('sim results', sim_results.shape)
        #print('y data', self.observed_data_y.shape)

        if np.any(np.isnan(sim_results)):
            return -np.inf

        # add them up separately because that seems to fail a lot less often
        # and I don't have correlations on the outputs anyways
        if independent:
            log_likelihood = 0
            for i in range(len(sim_results)):
                log_likelihood_i = scipy.stats.multivariate_normal.logpdf(
                    x=sim_results[i],
                    mean=self.observed_data_y[i],
                    cov=self.observed_data_y_uncertainties[i, i]
                )
                if log_likelihood_i == -np.inf:
                    log_likelihood_i = -1e80
            log_likelihood += log_likelihood_i
        else:
            log_likelihood = scipy.stats.multivariate_normal.logpdf(
                x=sim_results,
                mean=self.observed_data_y,
                cov=self.observed_data_y_uncertainties,
                allow_singular=True
            )
        # print('LL', log_likelihood)
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
        #print('logP=', log_posterior)
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


    def collect_samples(self, N_samples, N_walkers=None, discard=0.1):
        if N_walkers is None:
            N_walkers = int(2 * self.N_PARAMETERS)
        N_ENSEMBLE_STEPS = int(N_samples / N_walkers)

        # EXTENSION other ways to initialize? uniform distribution?
        if self.load_save_point:
            # figure out which chain this processor is associated with
            discard = 0  # already got through the burn-in
            # load existing chains and get last positions of walkers
            proc2chain = self.get_processor_to_chain_index(N_PROCESSORS, self.N_ZEUS_CHAINS)

            if RANK in proc2chain.keys():
                chain_index = proc2chain[RANK]
                chain_file = os.path.join(self.results_dir, f'chain_{chain_index}.npy')
                if not os.path.exists(chain_file):
                    raise FileNotFoundError(f"Cannot find existing chain file {chain_file} for loading save point.")
                chain = np.load(chain_file)
                # get last positions of walkers
                walker_start_points = chain[-1, :, :]
            else:
                # can we just ignore the non-master process walker start points?
                walker_start_points = np.zeros((N_walkers, self.N_PARAMETERS)) + np.nan
        else:
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
                maxiter=1e8,
                verbose=True
            )

            # run the zeus sampler
            zeus_sampler.run_mcmc(walker_start_points, N_ENSEMBLE_STEPS)
            print('done collecting samples')
            # save files
            chain = zeus_sampler.get_chain(flat=False, discard=discard)
            logPs = zeus_sampler.get_log_prob(flat=False, discard=discard)

            # note shape of chain and logPs
            if RANK == 0:
                print('single chain is ', chain.shape)
                print('single logP is', logPs.shape)

            # convert from processor index to chain index
            proc2chain = self.get_processor_to_chain_index(N_PROCESSORS, self.N_ZEUS_CHAINS)
            if RANK in proc2chain.keys():
                chain_index = proc2chain[RANK]
                chain_file = os.path.join(self.results_dir, f'chain_{chain_index}.npy')
                logP_file = os.path.join(self.results_dir, f'logPs_{chain_index}.npy')

                if self.load_save_point:
                    # need to append instead of overwrite
                    prev_chain = np.load(chain_file)
                    prev_logP = np.load(logP_file)


                    chain = np.vstack((prev_chain, chain))
                    logPs = np.vstack((prev_logP, logPs))

                np.save(chain_file, chain)
                np.save(logP_file, logPs)
            
            # maybe turn this off, or implement better logging options
            zeus_sampler.summary

    def compile_and_flatten_chains(self):
        # This function combines all the chain .npys into a single .npy.
        # Don't run this until you've finished all your sampling
        # order of operations is to run collect_samples lots of time
        # this produces 2 chains that grow longer and longer (not flattened)
        # then at the end they're combined into a single flattened, combined chain
        list_of_chains = []
        list_of_logPs = []
        for i in range(self.N_ZEUS_CHAINS):
            chain_file = os.path.join(self.results_dir, f'chain_{i}.npy')
            chain_i = np.load(chain_file)

            logPs_file = os.path.join(self.results_dir, f'logPs_{i}.npy')
            logP_i = np.load(logPs_file)

            assert chain_i.shape[2] == self.N_PARAMETERS
            list_of_chains.append(self.flatten_chain(chain_i))
            list_of_logPs.append(self.flatten_logP(logP_i))
        combined_chains = np.vstack(list_of_chains)
        combined_logPs = np.hstack(list_of_logPs)
        
        print('final chain shape is', combined_chains.shape)
        print('final logP shape is', combined_logPs.shape)

        np.save(os.path.join(self.results_dir, 'combined_chains.npy'), combined_chains)
        np.save(os.path.join(self.results_dir, 'combined_logPs.npy'), combined_logPs)

        

    def flatten_chain(self, chain):
        # https://github.com/minaskar/zeus/blob/1abdf08252a99e9aa186dcee414f559624b3bafd/zeus/samples.py#L90
        # Copied from zeus sample.flatten()
        assert chain.ndim == 3
        assert chain.shape[2] == self.N_PARAMETERS
        return chain.reshape((-1, chain.shape[2]), order='F')
        

    def flatten_logP(self, logP):
        assert logP.ndim == 2
        return logP.reshape((-1,), order='F')


    def make_all_plots(self):
        if RANK > 0:
            return

        flattened_combined_chain = np.load(os.path.join(self.results_dir, f'combined_chains.npy'))
        flattened_combined_logP = np.load(os.path.join(self.results_dir, f'combined_logPs.npy'))

        # Also load just the first chain for autocorrelation time
        chain0 = np.load(os.path.join(self.results_dir, 'chain_0.npy'))
        logP0 = np.load(os.path.join(self.results_dir, 'logPs_0.npy'))


        # find MAP
        MAP_index = np.argmin(np.abs(flattened_combined_logP - np.max(flattened_combined_logP)))
        print('MAP:', flattened_combined_logP[MAP_index], flattened_combined_chain[MAP_index, :])

        # maximalist plotting
        plotting.make_histograms(
            self.plot_dir,
            flattened_combined_chain,
            MAP=flattened_combined_chain[MAP_index, :],
            mean=np.nanmean(flattened_combined_chain, axis=0),
            initial=self.priors,
            parameter_names=self.parameter_names
        )

        # # corner plot / posterior scatter matrix
        # plotting.make_corner_plot(plot_dir, flattened_chain0)

        # corner plot / posterior scatter matrix
        plotting.make_posterior_scatter_matrix(self.plot_dir, flattened_combined_chain)

        # heat scatter matrix
        plotting.make_heat_scatter(
            self.plot_dir,
            flattened_combined_chain,
            MAP=flattened_combined_chain[MAP_index, :],
            mean=np.nanmean(flattened_combined_chain, axis=0),
            initial=self.priors,
            parameter_names=self.parameter_names
        )

        # maximal autocorrelation - needs full chain shape
        plotting.make_autocorrelation(self.plot_dir, chain0, parameter_names=self.parameter_names)
