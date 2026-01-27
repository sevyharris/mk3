# mk3
Markov chain Monte Carlo for microkinetic models

This is software to help set up parameter estimation for microkinetics models (simulated in Cantera) using MCMC
For now, it will just implement ESS using zeus-mcmc


# How to set up your own Cantera BPE run:

1. Implement sim wrapper.py
2. Fill in relevant sections of setup mk3 run
3. Run setup mk3 run


Pandas CSV is probably easier for experimental data than a yaml. human readable (beats .npy) but simpler formatting than yaml, more data/list/matrix friendly, but lets you have header names

Expected Files:
1. experiment.csv
    1. will have output names as optional header
    2. will get reshaped into a single array
    3. should probably have under 100 data points
2. experiment_u.csv
    1. choice of making this a list (of uncertainties AKA standard deviations) or a square matrix (cov)
3. prior.csv
4. prior_u.csv