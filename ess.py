# Let's sketch out generally how this will go

# imports
import os
import time
import numpy as np
import yaml
import zeus

import mpi4py.MPI
# probably have to handle MPI stuff here

RANK = mpi4py.MPI.COMM_WORLD.Get_rank()
np.random.seed(400 + RANK)

# load settings
# for now, these are constants just saved here, but later we'll let the user set them elsewhere
N_ZEUS_CHAINS = 2  # zeus paper recommends nchains=2 or 4
N_WALKERS = 8
N_SAMPLES = 100
N_ENSEMBLE_STEPS = int(N_SAMPLES / N_WALKERS)
N_PARAMETERS = 2

def simulationFunction(x,a,b): #here x is a scalar or an array and "a" and "b" are constants for the equation.
    #time.sleep(0.05)
    x =np.array(x)
    y = (x-a)**2 + b
    return y


observed_data_x_values = [600,1100,1400]
observed_data_y_values = [360500, 580500, 1620500]
observed_data_y_values_uncertainties = [200000, 300000, 200000]

priors = np.array([200, 500])
prior_uncertainties = np.array([100, 200])

# convert to covariance matrix if it's a list of std devs
if prior_uncertainties.ndim == 1:
    prior_uncertainties = np.diag(np.float_power(prior_uncertainties, 2.0))

def GetLogP(parameters):
    a_given = parameters[0]
    b_given = parameters[1]
    y = simulationFunction(observed_data_x_values, a_given, b_given)  #an alternatie simpler syntax to unpack the parameters would be: simulationFunction(x_values_for_data, *parametersArray)

    return y


if mpi4py.MPI.COMM_WORLD.Get_size() < 2:
    N_ZEUS_CHAINS = 1
    # raise ValueError('You gotta use MPI for now. At least 2 processes')

# set up the zeus sampler
with zeus.ChainManager(N_ZEUS_CHAINS) as cm:
    zeus_sampler = zeus.EnsembleSampler(
        N_WALKERS, N_PARAMETERS, logprob_fn=GetLogP, 
        pool=cm.get_pool, maxiter=1e6  # maxiter should be 1e6 instead of 1e4 ashi says
    )


def generateInitialPoints(numStartPoints=0, initialPointsDistributionType='uniform', relativeInitialDistributionSpread=1.0, numParameters = 0, centerPoint=None, gridsearchSamplingInterval = [], gridsearchSamplingRadii = []):
    #The initial points will be generated from a distribution based on the number of walkers and the distributions of the parameters.
    #The variable UserInput.std_prior has been populated with 1 sigma values, even for cases with uniform distributions.
    #The random generation at the front of the below expression is from the zeus example https://zeus-mcmc.readthedocs.io/en/latest/
    #The multiplication is based on the randn function using a sigma of one (which we then scale up) and then advising to add mu after: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randn.html
    #The actual numParameters cannot be 0. We just use 0 to mean not provided, in which case we pull it from the initial guess.
    #The arguments gridsearchSamplingInterval and gridsearchSamplingRadii are only for the distribution type 'grid', and correspond to the variables  gridsearchSamplingInterval = [], gridsearchSamplingRadii = [] inside getGridPermutations.
    if str(centerPoint).lower() == str(None).lower():
        centerPoint = priors
    if initialPointsDistributionType.lower() not in ['grid', 'uniform', 'identical', 'gaussian', 'astroidal', 'sobol', 'shell']:
        print("Warning: initialPointsDistributionType must be from: 'grid', 'uniform', 'identical', 'gaussian', 'astroidal', 'sobol', 'shell'.  A different choice was received and is not understood.  initialPointsDistributionType is being set as 'sobol'.")
        initialPointsDistributionType = 'sobol'
    #For a multi-start with a grid, our algorithm is completely different than other cases.
    if initialPointsDistributionType.lower() =='grid':
        gridPermutations, numPermutations = self.getGridPermutations(centerPoint, gridsearchSamplingInterval=gridsearchSamplingInterval, gridsearchSamplingRadii=gridsearchSamplingRadii)
        initialPoints = gridPermutations
    #Below lines are for non-grid cases.
    if numParameters == 0:
        numParameters = len(centerPoint)
    if numStartPoints == 0: #This is a deprecated line. The function was originally designed for making mcmc walkers and then was generalized.
        numStartPoints = self.mcmc_nwalkers
    if initialPointsDistributionType.lower() =='uniform':
        initialPointsFirstTerm = np.random.uniform(-2,2, [numStartPoints,numParameters]) #<-- this is from me, trying to remove bias. This way we get sampling from a uniform distribution from -2 standard deviations to +2 standard deviations. That way the sampling is over 95% of the prior and is (according to the prior) likely to include the HPD region.
    elif initialPointsDistributionType.lower()  == 'identical':
        initialPointsFirstTerm = np.zeros((numStartPoints, numParameters)) #Make the first term all zeros.
    elif initialPointsDistributionType.lower() =='gaussian':
        initialPointsFirstTerm = np.random.randn(numStartPoints, numParameters) #<--- this was from the zeus example. TODO: change this to rng.standard_normal
    elif (initialPointsDistributionType.lower() == 'astroidal') or (initialPointsDistributionType.lower() == 'shell'):
        # The idea is to create a hypercube around the origin then apply a power law factor.
        # This factor is set as the numParameters to create an interesting distribution for Euclidean distance that starts as a uniform distribution then decays by a power law if the exponent is the number of dimensions. 
        from scipy.stats import qmc
        from warnings import catch_warnings, simplefilter #used to suppress warnings when sobol samples are not base2.
        # A sobol object has to be created to then extract points from the object.
        # The scramble (Owen Scramble) is always True. This option helps convergence and creates a more unbiased sampling.
        sobol_object = qmc.Sobol(d=numParameters, scramble=True)
        with catch_warnings():
            simplefilter("ignore")
            sobol_samples = sobol_object.random(numStartPoints)
        # now we must translate the sequence (from range(0,1) to range(-2,2)). This is analagous to the way we get sampling from a uniform distribution from -2 standard deviations to +2 standard deviations.
        initialPointsFirstTerm = -1 + 2*sobol_samples
        # This section assures that positive and negative values are generated.
        # create mapping scheme of negative values, then make matrix completely positive, apply negatives back later
        neg_map = np.ones((numStartPoints,numParameters), dtype=int)
        neg_map[initialPointsFirstTerm < 0] = -1
        initialPointsFirstTerm = np.abs(initialPointsFirstTerm)
        if initialPointsDistributionType.lower() == 'astroidal':
            initialPointsFirstTerm = initialPointsFirstTerm**numParameters
        elif initialPointsDistributionType.lower() == 'shell':
            initialPointsFirstTerm = initialPointsFirstTerm**(1/numParameters)
        initialPointsFirstTerm = neg_map*initialPointsFirstTerm
        # Apply a proportional factor of 2 to get bounds of 2 sigma. This is analagous to the way we get sampling from a uniform distribution from -2 standard deviations to +2 standard deviations.
        initialPointsFirstTerm *= 2
    elif initialPointsDistributionType.lower() == 'sobol':
        from scipy.stats import qmc
        from warnings import catch_warnings, simplefilter #used to suppress warnings when sobol samples are not base2.
        # A sobol object has to be created to then extract points from the object.
        # The scramble (Owen Scramble) is always True. This option helps convergence and creates a more unbiased sampling.
        sobol_object = qmc.Sobol(d=numParameters, scramble=True)
        with catch_warnings():
            simplefilter("ignore")
            sobol_samples = sobol_object.random(numStartPoints)
        # now we must translate the sequence (from range(0,1) to range(-2,2)). This is analagous to the way we get sampling from a uniform distribution from -2 standard deviations to +2 standard deviations.
        initialPointsFirstTerm = -2 + 4*sobol_samples
    if initialPointsDistributionType !='grid':
        initialPoints = relativeInitialDistributionSpread*initialPointsFirstTerm*np.sqrt(prior_uncertainties[0][0]) + centerPoint
    return initialPoints

walker_start_points = generateInitialPoints(initialPointsDistributionType='uniform', numStartPoints=N_WALKERS,relativeInitialDistributionSpread=0.866) 
# walker_start_points = np.random.multivariate_normal(priors, prior_uncertainties / 10.0, size=N_WALKERS, check_valid='warn', tol=1e-8)  # probably want uncertainties to be lower than usual
    
print(walker_start_points)

# run the zeus sampler
zeus_sampler.run_mcmc(walker_start_points, N_ENSEMBLE_STEPS)

# save files

# do analysis

# make plots