# A script to set up a fresh BPE run

import os
import shutil

import cantera as ct
import yaml
import numpy as np
import pandas as pd

import scipy.interpolate


def get_i_thing(ref_composition, phase):
    """
    Get index of a requested Cantera species
    ref_composition is a dictionary of the atom counts
    phase is the cantera phase in which the species is stored
    """
    for i in range(phase.n_species):
        if phase.species()[i].composition == ref_composition:
            return i
    assert False


# ------------------------------ Specify Mechanism  --------------------------
# mech_yaml = '/home/harris.se/chem_annotated_noCH4X.yaml'
mech_yaml = '/home/moon/chem_annotated_noCH4X.yaml'
# ----------------------------------------------------------------------------


# ------------- Specify working directory for your next run-------------------
# working_dir is where you want to set up your next run
# working_dir = '/scratch/harris.se/guassian_scratch/my_kinetics_bpe0'
# working_dir = '/home/moon/mk3/example_cantera'
working_dir = '/home/moon/mk3/ct_example'
# ----------------------------------------------------------------------------


# In this house we don't overwrite existing files
assert not os.path.exists(working_dir), f"Working directory {working_dir} already exists. Please choose a different one or delete it."
os.makedirs(working_dir)

experimental_yaml_file = os.path.join(working_dir, 'experiment.yaml')
prior_yaml_file = os.path.join(working_dir, 'prior.yaml')

UNCERTAINTY_REPO = os.environ['UNCERTAINTY_REPO']

# TODO - move default BPE scripts in mk3 repo
base_run_bpe_py_script = os.path.join(UNCERTAINTY_REPO, 'bpe', 'simulation', 'run_bpe.py')
base_run_bpe_sh_script = os.path.join(UNCERTAINTY_REPO, 'bpe', 'simulation', 'run_bpe.sh')
shutil.copy(base_run_bpe_py_script, os.path.join(working_dir, 'run_bpe.py'))
shutil.copy(base_run_bpe_sh_script, os.path.join(working_dir, 'run_bpe.sh'))
shutil.copy(mech_yaml, os.path.join(working_dir, 'chem_annotated.yaml'))


# load the mechanism yaml to get the species and reaction information
gas = ct.Solution(mech_yaml)
surf = ct.Interface(mech_yaml, "surface1", [gas])

# Get indices of key species
i_Ar = get_i_thing({'Ar': 1.0}, gas)
i_CH4 = get_i_thing({'C': 1.0, 'H': 4.0}, gas)
i_O2 = get_i_thing({'O': 2.0}, gas)
i_CO2 = get_i_thing({'C': 1.0, 'O': 2.0}, gas)
i_H2O = get_i_thing({'H': 2.0, 'O': 1.0}, gas)
i_H2 = get_i_thing({'H': 2.0}, gas)
i_CO = get_i_thing({'C': 1.0, 'O': 1.0}, gas)
i_C2H4 = get_i_thing({'C': 2.0, 'H': 4.0}, gas)
i_X = get_i_thing({'X': 1.0}, surf)
i_OX = get_i_thing({'X': 1.0, 'O': 1.0}, surf)
i_CX = get_i_thing({'X': 1.0, 'C': 1.0}, surf)
i_CO2X = get_i_thing({'X': 1.0, 'C': 1.0, 'O': 2.0}, surf)
i_COX = get_i_thing({'X': 1.0, 'C': 1.0, 'O': 1.0}, surf)
i_HX = get_i_thing({'X': 1.0, 'H': 1.0}, surf)
i_CH3X = get_i_thing({'X': 1.0, 'H': 3.0, 'C': 1.0}, surf)
i_H2OX = get_i_thing({'X': 1.0, 'H': 2.0, 'O': 1.0}, surf)
i_H2X = get_i_thing({'X': 1.0, 'H': 2.0}, surf)

# -------------- Pick the species/reactions you want to vary here -------- #
# TODO let user vary gas species/reactions
my_species_indices = []
my_reaction_indices = [0, 1, 2, 3, 4, 5, 13, 14]
# ------------------------------------------------------------------------ #
for i in my_species_indices:
    print(f'Variable species {i}:', surf.species_names[i])
for i in my_reaction_indices:
    print(f'Variable reaction {i}:', surf.reactions()[i].equation)


# ------------------------- Set the experimental error ------------------- #
experimental_error = 0.05
MIN_EXP_ERROR = 1e-8


# TODO make simulation separate from number of experimental points checked
N_dist_pts = 20
DIST_START = 0.000
DIST_END = 0.010
dist_array = np.linspace(DIST_START, DIST_END, N_dist_pts)


# -------------------------------- Experimental Info --------------------------------
# Load the experimental data and save relevant info in experiment.yaml
pt_data_file = os.path.join(UNCERTAINTY_REPO, 'cpox_pt', 'horn_data', 'pt_profiles_smooth.csv')
df = pd.read_csv(pt_data_file)
distances = (df['Distance (mm)'] - 10.0) / 1000.0  # ignore the 10mm of no/catalyst space

# Make interpolation of temperature
exp_Ts = df['Temperature (K)']
f_T = scipy.interpolate.interp1d(distances, exp_Ts, fill_value='extrapolate')
Ts = f_T(dist_array)

species_headers = df.columns[2:]
experimental_data = {
    'Distance (m)': dist_array,
    'Temperature (K)': Ts
}

for header in species_headers:
    exp_values = df[header]
    f = scipy.interpolate.interp1d(distances, exp_values, fill_value='extrapolate')
    experimental_data[header] = f(dist_array)
    
for header in species_headers:
    uncertainty_header = 'Uncertainty ' + header
    experimental_data[uncertainty_header] = experimental_data[header] * experimental_error
    experimental_data[uncertainty_header][experimental_data[uncertainty_header] < MIN_EXP_ERROR] = MIN_EXP_ERROR

    
# Convert from numpy arrays into lists
for key in experimental_data.keys():
    experimental_data[key] = experimental_data[key].tolist()

    
out_gas_indices = [i_CH4, i_O2, i_H2O, i_H2, i_CO, i_CO2]
experimental_data['out_gas_indices'] = out_gas_indices

out_gas_names = ['CH4 (mol/min)', 'O2 (mol/min)', 'H2O (mol/min)', 'H2 (mol/min)', 'CO (mol/min)', 'CO2 (mol/min)']
experimental_data['out_gas_names'] = out_gas_names
    
with open(experimental_yaml_file, 'w') as outfile:
    yaml.dump(experimental_data, outfile, default_flow_style=False)
# with open(experimental_yaml_file) as f:
#     data = yaml.safe_load(f)



# -------------------------------- Prior Info --------------------------------
# TODO - grab uncertainties from RMG uncertainty tool
prior_data = {}

n_sp = len(my_species_indices)
n_rxn = len(my_reaction_indices)

eV_to_j_mol = 96485

std_dev_thermo = 0.3 * eV_to_j_mol
std_dev_log10k = 3.0  # factor of 10, should translate into multiplier of 0.1 - 10


prior_cov_J_mol = np.identity(n_sp + n_rxn) * np.float_power(std_dev_thermo, 2.0)  # but this should be squared
for i in range(n_sp, n_sp + n_rxn):
    prior_cov_J_mol[i, i] = std_dev_log10k  # kinetics part of cov matrix
    
for i in my_species_indices:
    key = surf.species_names[i]
    prior_data[key] = 0.0
    
for i in my_reaction_indices:
    key = surf.reactions()[i].equation
    prior_data[key] = 0.0

    

# flatten prior covariance matrix into list of lists
    
prior_data['cov_J2_mol2'] = prior_cov_J_mol.tolist()

prior_data['species_indices'] = my_species_indices
prior_data['species_names'] = [surf.species_names[i] for i in my_species_indices]

prior_data['reaction_indices'] = my_reaction_indices
prior_data['reaction_equations'] = [surf.reactions()[i].equation for i in my_reaction_indices]
with open(prior_yaml_file, 'w') as outfile:
    yaml.dump(prior_data, outfile, default_flow_style=False)
# with open(prior_yaml_file) as f:
#     data = yaml.safe_load(f)
