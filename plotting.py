import os
import matplotlib.pyplot as plt


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

        plt.savefig(outfile)
