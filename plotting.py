import os
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib
import zeus
import zeus.autocorr
import numpy as np
import scipy.interpolate
import pandas as pd

MAP_COLOR = 'r'
INITIAL_COLOR = '#00A5DF'
MEAN_COLOR = 'k'


def make_histograms(outdir, samples, MAP=None, mean=None, initial=None, parameter_names=None, **kwargs):
    """
    Generate histograms for each parameter

    outfile is the file path for the histogram images
    samples is the NxP array of samples, with P parameters and N samples
    MAP is an array of parameter values where likelihood is highest
    parameter names is a list of the parameter names
    kwargs are passed into the histogram if you want to control the plotting settings
    """
    if parameter_names is not None:
        assert len(parameter_names) == samples.shape[1]
    if MAP is not None:
        assert len(MAP) == samples.shape[1]
    if mean is not None:
        assert len(mean) == samples.shape[1]
    if initial is not None:
        assert len(initial) == samples.shape[1]

    for i in range(samples.shape[1]):

        param_name = f'p{i}'
        if parameter_names is not None:
            param_name = parameter_names[i]
        outfile = os.path.join(outdir, f'hist_{param_name}.png')
        
        plt.figure()
        plt.hist(samples[:, i], **kwargs)

        if MAP is not None:
            plt.axvline(x=MAP[i], color=MAP_COLOR, label='MAP')
        if mean is not None:
            plt.axvline(x=mean[i], color=MEAN_COLOR, label='Mean')
        if initial is not None:
            plt.axvline(x=initial[i], color=INITIAL_COLOR, label='Initial')

        plt.xlabel(param_name)
        plt.ylabel('Count')
        
        label_count = len(plt.gca().get_legend_handles_labels()[0])
        if label_count > 0:
            plt.legend()

        plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def make_corner_plot(outdir, samples):
    # this one's the same as the zeus corner plot
    # zeus-mcmc.readthedocs.io/en/latest/notebooks/normal_distribution.html
    fig, axes = zeus.cornerplot(samples[::100], size=(16,16))
    outfile = os.path.join(outdir, 'cornerplot.png')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


def make_posterior_scatter_matrix(outdir, samples, parameter_names=None):
    # convert to pandas dataframe - the PEUQSE way
    if parameter_names is None:
        parameter_names = [f'p{i}' for i in range(samples.shape[1])]
        assert len(parameter_names) == samples.shape[1]
    else:
        parameter_names

    posterior_df = pd.DataFrame(samples, columns=parameter_names)
    pd.plotting.scatter_matrix(posterior_df)
    outfile = os.path.join(outdir, 'posterior_scatter_matrix.png')

    plt.savefig(outfile, bbox_inches='tight')
    plt.close()


def make_heat_scatter(outdir, samples, MAP=None, mean=None, initial=None, parameter_names=None, **kwargs):
    # heat scatter for each pair of parameters
    # function from PEUQSE's density_scatter
    # colored by density https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib/53865762#53865762

    if parameter_names is not None:
        assert len(parameter_names) == samples.shape[1]
    if MAP is not None:
        assert len(MAP) == samples.shape[1]
    if mean is not None:
        assert len(mean) == samples.shape[1]
    if initial is not None:
        assert len(initial) == samples.shape[1]

    # go through the pairs of parameters
    for i in range(1, samples.shape[1]):
        for j in range(i):
            plt.figure()
            x_name = f'p{i}'
            y_name = f'p{j}'
            if parameter_names is not None:
                x_name = parameter_names[i]
                y_name = parameter_names[j]

            x = samples[:, i]
            y = samples[:, j]

            bin_density, x_edges, y_edges = np.histogram2d(x, y, density=True)
            x_grid = 0.5 * (x_edges[:-1] + x_edges[1:])  # take average because interpn takes in 1 fewer point than x_edges generates
            y_grid = 0.5 * (y_edges[:-1] + y_edges[1:])
            data_combined = np.array([(x[i], y[i]) for i in range(len(x))])
            sample_density = scipy.interpolate.interpn((x_grid, y_grid), bin_density, data_combined, bounds_error=False)

            # plot all data even if interpolation fails
            sample_density[np.isnan(sample_density)] = 0.0

            # plot densest points on top, but maybe comment this out if it takes too long to
            idx = sample_density.argsort()
            x, y, sample_density = x[idx], y[idx], sample_density[idx]


            plt.scatter(x, y, c=sample_density, s=0.8, **kwargs)
            ax = plt.gca()
            ax.set_box_aspect(1.0) 

            if MAP is not None:
                plt.scatter(MAP[i], MAP[j], c=MAP_COLOR, marker='x', label='MAP')
            if mean is not None:
                plt.scatter(mean[i], mean[j], c=MEAN_COLOR, marker='x', label='Mean')
            if initial is not None:
                plt.scatter(initial[i], initial[j], c=INITIAL_COLOR, marker='x', label='Initial')

            label_count = len(plt.gca().get_legend_handles_labels()[0])
            if label_count > 0:
                plt.legend()

            plt.xlabel(x_name)
            plt.ylabel(y_name)
            normalized_colorscale = matplotlib.colors.Normalize(vmin=np.min(sample_density), vmax=np.max(sample_density))
            cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normalized_colorscale), ax=ax)
            cbar.ax.set_ylabel('Density')

            outfile = os.path.join(outdir, f'heat_scatter_{y_name}_{x_name}.png')
            plt.savefig(outfile, bbox_inches='tight')
            plt.close()

def make_autocorrelation(outdir, chain, parameter_names=None):
    # chain should be N_samples x N_walkers x N_parameters, or NxWxP
    N_taus = 15
    N_samples, N_walkers, N_parameters = chain.shape
    if parameter_names is not None:
        assert len(parameter_names) == N_parameters

    # estimate autocorrelation time at increasing # samples that is uniformly spaced on log scale
    window_indices = np.logspace(0, np.log10(N_samples), N_taus).astype(int)

    taus = np.zeros((N_taus, N_parameters)) 
    for i, w in enumerate(window_indices):
        taus[i, :] = zeus.autocorr.AutoCorrTime(chain[:w, :, :])

    # make individual autocorrelation plots
    for i in range(N_parameters):
        plt.figure()
        plt.loglog(window_indices, taus[:, i], marker='o', label=r'Estimated $\tau$')
        plt.plot(window_indices, window_indices / 50.0, linestyle='dashed', color='black', label=r'$\tau$=N/50')

        param_name = f'p{i}'
        if parameter_names is not None:
            param_name = parameter_names[i]
        outfile = os.path.join(outdir, f'autocorr_{param_name}.png')
        plt.xlabel('N Samples')
        plt.ylabel(param_name + ' -- ' + r'Estimated $\tau$')
        plt.legend()
        plt.savefig(outfile, bbox_inches='tight')
        plt.close()

    # Make a combined autocorrelation plot
    plt.figure()
    for i in range(N_parameters):
        param_name = f'p{i}'
        if parameter_names is not None:
            param_name = parameter_names[i]
        plt.loglog(window_indices, taus[:, i], marker='o', label=param_name)
    
    plt.plot(window_indices, window_indices / 50.0, linestyle='dashed', color='black', label=r'$\tau$=N/50')
    plt.xlabel('N Samples')
    plt.ylabel(r'Estimated $\tau$')
    plt.legend()
    outfile = os.path.join(outdir, f'combined_autocorr.png')
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
