# Let's sketch out generally how this will go

# imports
import os
import numpy as np
import dill
import zeus
import zeus.parallel
import mpi4py
import scipy.stats
import plotting


global MPI

from mpi4py import MPI as _MPI

_MPI.pickle.__init__(dill.dumps, dill.loads, dill.HIGHEST_PROTOCOL)
MPI = _MPI

# probably have to handle MPI stuff here
N_PROCESSORS = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
np.random.seed(400 + RANK)


class BPEstimator():
    """
    A wrapper class for running zeus ensemble slice sampling

    =================== =========================== ============================
    Attribute           Type                        Description
    =================== =========================== ============================
    `index`             :class:`int`                A unique nonnegative integer index
    `label`             ``str``                     A descriptive string label
    `reactants`         :class:`list`               The reactant species (as :class:`Species` objects)
    `products`          :class:`list`               The product species (as :class:`Species` objects)
    'specific_collider'  :class:`Species`            The collider species (as a :class:`Species` object)
    `kinetics`          :class:`KineticsModel`      The kinetics model to use for the reaction
    `network_kinetics`  :class:`Arrhenius`          The kinetics model to use for PDep network exploration if the `kinetics` attribute is :class:PDepKineticsModel:
    `reversible`        ``bool``                    ``True`` if the reaction is reversible, ``False`` if not
    `transition_state`   :class:`TransitionState`    The transition state
    `duplicate`         ``bool``                    ``True`` if the reaction is known to be a duplicate, ``False`` if not
    `degeneracy`        :class:`double`             The reaction path degeneracy for the reaction
    `pairs`             ``list``                    Reactant-product pairings to use in converting reaction flux to species flux
    `allow_pdep_route`  ``bool``                    ``True`` if the reaction has an additional PDep pathway, ``False`` if not (by default), used for LibraryReactions
    `elementary_high_p` ``bool``                    If ``True``, pressure dependent kinetics will be generated (relevant only for unimolecular library reactions)
                                                    If ``False`` (by default), this library reaction will not be explored.
                                                    Only unimolecular library reactions with high pressure limit kinetics should be flagged (not if the kinetics were measured at some relatively low pressure)
    `comment`           ``str``                     A description of the reaction source (optional)
    `is_forward`        ``bool``                    Indicates if the reaction was generated in the forward (true) or reverse (false)
    `rank`              ``int``                     Integer indicating the accuracy of the kinetics for this reaction
    =================== =========================== ============================

    """
    def __init__(
        self,
        simulation_fn,
        priors,
        prior_uncertainties,
        observed_data_x,
        observed_data_y,
        observed_data_y_uncertainties,
        results_dir=None
    ):
        self.simulation_fn = simulation_fn
        self.priors = priors
        self.prior_uncertainties = prior_uncertainties
        self.observed_data_x = observed_data_x
        self.observed_data_y = observed_data_y
        self.observed_data_y_uncertainties = observed_data_y_uncertainties
        self.sampler = None

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

        self.results_dir = './'
        if results_dir is not None:
            self.results_dir = results_dir
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)

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
            self.sampler = zeus.EnsembleSampler(
                N_walkers,
                self.N_PARAMETERS,
                logprob_fn=self.get_log_posterior, 
                pool=cm.get_pool,
                maxiter=1e6
            )

            # run the zeus sampler
            self.sampler.run_mcmc(walker_start_points, N_ENSEMBLE_STEPS)

            # maybe turn this off, or implement better logging options
            self.sampler.summary

            # save files
            chain = self.sampler.get_chain(flat=False, discard=discard)
            logPs = self.sampler.get_log_prob(flat=False, discard=discard)

            # convert from processor index to chain index
            proc2chain = self.get_processor_to_chain_index(N_PROCESSORS, self.N_ZEUS_CHAINS)
            if RANK in proc2chain.keys():
                chain_index = proc2chain[RANK]
                np.save(os.path.join(self.results_dir, f'chain_{chain_index}.npy'), chain)
                np.save(os.path.join(self.results_dir, f'logPs_{chain_index}.npy'), logPs)

    
# # print(samples.shape)

# # do analysis
# chain0 = np.load('chain_0.npy')
# print(chain0.shape)
# logP0 = np.load('logPs_0.npy')
# print(logP0.shape)

# MAP_index = np.argmin(np.abs(logP0 - np.max(logP0)))
# print('MAP:', logP0[MAP_index], chain0[MAP_index, :])


# # Plotting

# outdir = '/home/moon/mk3/'

# # maximalist plotting
# # plotting.make_histograms(
# #     outdir,
# #     chain0,
# #     MAP=chain0[MAP_index, :],
# #     mean=np.nanmean(chain0, axis=0),
# #     initial=priors,
# #     parameter_names=['a', 'b']
# # )

# # minimal hist
# plotting.make_histograms(outdir, chain0)


# # corner plot / posterior scatter matrix
# plotting.make_corner_plot(outdir, chain0)

# # corner plot / posterior scatter matrix
# plotting.make_posterior_scatter_matrix(outdir, chain0)

# # heat scatter matrix
# # maximal
# plotting.make_heat_scatter(
#     outdir,
#     chain0,
#     MAP=chain0[MAP_index, :],
#     mean=np.nanmean(chain0, axis=0),
#     initial=priors,
#     parameter_names=['a', 'b']
# )

# # minimal
# # plotting.make_heat_scatter(outdir, chain0)

# # minimal autocorrelation
# # plotting.make_autocorrelation(outdir, full_chain)

# # maximal autocorrelation
# plotting.make_autocorrelation(outdir, full_chain, parameter_names=['a', 'b'])
