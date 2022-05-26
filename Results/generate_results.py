import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import softmax
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

head_direction = 'max'

rescale = False # Required for head_direction = 'mean' to work (else negative probabilities can occur), optional otherwise
softmax_scaling = False

plot_samples = True
plot_timeline = True
plot_error = False

save_figures = False

clamp_sample_count = False

dataset_root_path = ' ' # Point to folder containing dataset folders

representations_root_path = ' ' # Point to folder containing representations folders

benchmark_root_path = ' ' # Optional: Point to folder containing representations from benchmark model, else make equal to representations_root_path

results_root_path = ' ' # Point to output filepath. If generating figures, make sure this matches

representations_label = 'PredNet' # Or VAE/CNN if generating for those

benchmark_label = 'Full Set Idiothetic'

sample_filename = 'sample_multiprednet'

timeline_filename = 'timeline_multiprednet'

def evaluate(network_data_path, multiprednet_data_path, benchmark_data_path, results_save_path, number_of_samples = 500, head_direction = head_direction, plot_samples = False, plot_timeline = False, plot_error = False):

    network_spike_output = np.load(network_data_path + '/networkOutput.npy').T
    multiprednet_output = np.load(multiprednet_data_path + '/reconstructions_head_direction.npy')
    benchmark_output = np.load(benchmark_data_path + '/reconstructions_head_direction.npy') # Change if using a benchmark, keeping the same does no harm

    if clamp_sample_count is False:

        network_spike_output = network_spike_output[:number_of_samples]
        multiprednet_output = multiprednet_output[:number_of_samples]
        benchmark_output = benchmark_output[:number_of_samples]

    else:

        minimum_sample_count = min(network_spike_output.shape[0], multiprednet_output.shape[0], benchmark_output.shape[0])

        network_spike_output = network_spike_output[:minimum_sample_count]
        multiprednet_output = multiprednet_output[:minimum_sample_count]
        benchmark_output = benchmark_output[:minimum_sample_count]

    print(network_spike_output.shape)
    print(multiprednet_output.shape)
    print(benchmark_output.shape)

    network_head_direction = np.argmax(network_spike_output, axis = 1)

    if rescale is True:

        multiprednet_output = minmax_scale(multiprednet_output, axis = 1)
        benchmark_output = minmax_scale(benchmark_output, axis = 1)

    if softmax_scaling is True:

        multiprednet_output = softmax(multiprednet_output, axis = 1)
        benchmark_output = softmax(benchmark_output, axis = 1)

    # Find max of multiprednet

    indices = np.arange(1, multiprednet_output.shape[1]+1)

    indices = np.resize(indices, multiprednet_output.shape)

    weighted_probabilities = np.multiply(indices, multiprednet_output)

    multiprednet_head_direction = np.round(np.mean(weighted_probabilities, axis = 1))

    multiprednet_head_direction_mean = multiprednet_head_direction

    multiprednet_head_direction = np.argmax(multiprednet_output, axis = 1)

    multiprednet_head_direction_max = multiprednet_head_direction

    print(multiprednet_head_direction_max.shape)

    # Find max of benchmark

    indices = np.arange(1, benchmark_output.shape[1]+1)

    indices = np.resize(indices, benchmark_output.shape)

    weighted_probabilities = np.multiply(indices, benchmark_output)

    benchmark_head_direction = np.round(np.mean(weighted_probabilities, axis = 1))

    benchmark_head_direction_mean = benchmark_head_direction

    benchmark_head_direction = np.argmax(benchmark_output, axis = 1)

    benchmark_head_direction_max = benchmark_head_direction

    print(benchmark_head_direction_max.shape)

    if plot_samples:

        fig, ax = plt.subplots(1, 1)

        sample = np.arange(1, multiprednet_output.shape[0]+2)[41]#[np.random.randint(1, multiprednet_output.shape[1]+2)]

        ax.plot(multiprednet_output[sample], color = 'red', label = representations_label)
        ax.plot(benchmark_output[sample], color = 'blue', label = benchmark_label)
        ax.vlines(network_head_direction[sample], np.min(multiprednet_output[sample]), np.max(multiprednet_output[sample])+0.2, color = 'black', label = 'Ground Truth Head Direction')
        ax.vlines(multiprednet_head_direction_max[sample], np.min(multiprednet_output[sample]), np.max(multiprednet_output[sample])+0.1, color = 'red', label = representations_label)
        ax.vlines(benchmark_head_direction_max[sample], np.min(benchmark_output[sample]), np.max(benchmark_output[sample]), color = 'blue', label = benchmark_label)
        plt.title("Head Direction: Multiprednet vs Ground Truth")
        plt.xlabel("Head Direction")
        plt.ylabel("Pseudo-probability")
        ax.legend()

        if save_figures is True:

            plt.savefig(results_save_path + '{}.png'.format(sample_filename))

    if plot_timeline:

        fig, ax = plt.subplots(1, 1)
        
        ax.plot(network_head_direction, color = 'black', label = 'Ground Truth')
        ax.plot(multiprednet_head_direction, color = 'red', label = representations_label)
        #ax.plot(benchmark_head_direction, color = 'blue', label = benchmark_label)
        plt.title("Head Direction Reconstruction")
        plt.xlabel("Sample")
        plt.ylabel("Cell")
        ax.legend()

        if save_figures is True:

            plt.savefig(results_save_path + '{}.png'.format(timeline_filename))

    multiprednet_error = []
    mmvae_error = []

    # Since head direction wraps around, find the error in both directions and take the minimum

    for predicted, actual in zip(multiprednet_head_direction, network_head_direction):

        error_forward = np.abs((predicted - actual) % 20)
        error_backward = np.abs((actual - predicted) % 20)

        multiprednet_error.append(min(error_forward, error_backward))

    for predicted, actual in zip(benchmark_head_direction, network_head_direction):

        error_forward = np.abs((predicted - actual) % 20)
        error_backward = np.abs((actual - predicted) % 20)

        mmvae_error.append(min(error_forward, error_backward))

    if plot_error:

        fig, ax = plt.subplots(1, 1)
        
        ax.plot(error, color = 'red', label = 'Error')
        ax.hlines(np.mean(np.array(error)), 0, 100, label = 'Mean Error')
        ax.legend()

        if save_figures is True:

            plt.savefig(results_save_path + 'error.png')

    return multiprednet_error, mmvae_error

if True:
    dataset = ' ' # Point to dataset folder to generate results
    
# Alternatively: "for dataset in (<comma-seperated folder strings>):" can be used to generate multiple

    print("Testset {}; Visual Only".format(dataset))

    error = evaluate(   network_data_path = dataset_root_path + '{}'.format(dataset),
                        multiprednet_data_path = representations_root_path + '{}/visual'.format(dataset),
                        benchmark_data_path = benchmark_root_path + '{}/visual'.format(dataset),
                        results_save_path = results_root_path + '{}/visual/'.format(dataset),
                        head_direction = head_direction,
                        plot_samples = plot_samples,
                        plot_timeline = plot_timeline,
                        plot_error = plot_error)

    total_error_multiprednet = np.sum(error[0])
    mean_error_multiprednet = np.mean(error[0])

    total_error_mmvae = np.sum(error[1])
    mean_error_mmvae = np.mean(error[1])

    print("Mean MultiPredNet Error = {}, Total MultiPredNet Error = {}\n".format(mean_error_multiprednet, total_error_multiprednet))
    print("Mean Benchmark Error = {}, Total Benchmark Error = {}\n".format(mean_error_mmvae, total_error_mmvae))

    if plot_samples is True or plot_timeline is True or plot_error is True:

        plt.show()
