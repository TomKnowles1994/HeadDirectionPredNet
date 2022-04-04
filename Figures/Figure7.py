import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, wasserstein_distance, sem
from scipy.special import softmax, kl_div
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error

mode = 'ground_truth_to_prediction'

mean_random = True

#full_colour = '#63F500'
#single_colour = '#09D4F5'
#ideo_colour = '#F70C7D'

#full_colour = '#4CBF00'
#single_colour = '#08A9C2'
#ideo_colour = '#BF0A61'

#full_colour = None
#single_colour = None
#ideo_colour = None

pcn_colour = '#63F500'
vae_colour = '#09D4F5'
cnn_colour = '#F70C7D'#031BFF'

#pcn_colour = '#F53313'#F56700'#F56000'
#vae_colour = '#3A78CF'#00F5A5'
#cnn_colour = '#8925BD'#3200FA'

#pcn_colour = '#713BF5'#F56000'
#vae_colour = '#12B325'#00F5A5'
#cnn_colour = '#FC9D3F'#3200FA'

#bar_width = 0.2#0.18

def load_datasets(file_names, max_samples, clip = None, raise_floor = False, transpose = False):

    datasets = []

    for dataset in file_names:

        data = np.load(dataset)

        if transpose:

            data = data.T

        data = data[:max_samples]

        data[np.where(data == 0)] = 0.0001

        if clip is not None:

            data[np.where(data < clip)] = 0.0001

        if raise_floor is True:

            data = data + np.abs(np.min(data)) + 0.0001

        datasets.append(data)

    return datasets

def convert_to_laplacians(reconstructions_list):

    laplacians = []

    for reconstructions in reconstructions_list:

        sharpness = 20

        N = 180

        gaussian_width = reconstructions.shape[1]

        gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2)) # Used to be up to (gaussian_width//2)+1 (adjusts for odd numbers)

        gaussian_block = np.resize(gaussian_range, new_shape = (reconstructions.shape))

        max_locations = np.argmax(reconstructions, axis = 1)

        pose_gaussians = laplace(0, gaussian_width//sharpness)

        zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

        shifted_gaussians = np.empty_like(reconstructions)

        shifted = np.roll(zeroed_gaussians[0], max_locations[0]-(N//2)) # Used to be hard-coded to -31 i.e. -63//2

        for index in range(len(max_locations)):
            shifted = np.roll(zeroed_gaussians[index,:], max_locations[index]-(N//2))
            shifted_gaussians[index,:] = shifted

        laplacians.append(shifted_gaussians)

    return laplacians

def calculate_divergence(prediction_list, laplacian_list, metric):

    total_distances_list = []

    if metric is 'wasserstein':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            wasserstein_distances = []

            #print(predictions.shape)

            for prediction, laplacian in zip(predictions, laplacians):

                #wasserstein_forward = wasserstein_distance(prediction, laplacian) % 180
                #wasserstein_backward = wasserstein_distance(laplacian, prediction) % 180

                #wasserstein_distances.append(np.min([wasserstein_forward, wasserstein_backward]))

                wasserstein_distances.append(wasserstein_distance(prediction, laplacian))

            total_distances_list.append(np.mean(wasserstein_distances))

    elif metric is 'kl':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            kl_divergences = []

            for prediction, laplacian in zip(predictions, laplacians):

                #print(prediction.shape)
                #print(laplacian.shape)

                kl_divergences.append(kl_div(prediction, laplacian))

            total_distances_list.append(np.mean(kl_divergences))

    elif metric is 'js':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            js_distances = []

            for prediction, laplacian in zip(predictions, laplacians):

                js_distances.append(jensenshannon(prediction, laplacian))

            total_distances_list.append(np.mean(js_distances))

    elif metric is 'rmse':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            rmses = []

            for prediction, laplacian in zip(predictions, laplacians):

                rmses.append(mean_squared_error(prediction, laplacian, squared = False))

            total_distances_list.append(np.mean(rmses))

    return total_distances_list

def distribution_fudge(prediction_list):

    for i in range(len(prediction_list)):

        prediction_list[i] = prediction_list[i] - np.min(prediction_list[i]) + 0.000000001

    return prediction_list

datasets_root_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_'

representations_root_path = 'C:/Users/Thomas/Downloads/HBP/representations/NRP/whiskeye_head_direction_'

results_root_path = 'C:/Users/Thomas/Downloads/HBP/head_direction_evaluation/whiskeye_head_direction_'

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames_pcn = [      representations_root_path + 'rotating_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5/visual/reconstructions_head_direction.npy']

single_representations_filenames_pcn = [    representations_root_path + 'rotating_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5_single/visual/reconstructions_head_direction.npy']

ideo_representations_filenames_pcn = [      representations_root_path + 'rotating_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5_ideo/visual/reconstructions_head_direction.npy']


full_representations_predictions = load_datasets(full_representations_filenames_pcn, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames_pcn, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames_pcn, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = 'row')

fig.set_figheight(4)
fig.set_figwidth(10)

#fig.tight_layout()

### Jensen-Shannon ###

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon_pcn = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon_pcn = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon_pcn = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon_pcn = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon_pcn = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon_pcn = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

### ConvNet Reconstruction Quality ###

representations_root_path = 'C:/Users/Thomas/Downloads/HBP/representations/NRP/whiskeye_head_direction_'

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames_vae = [      representations_root_path + 'rotating_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_full_mmvae/visual/reconstructions_multimodal_pose.npy']

single_representations_filenames_vae = [    representations_root_path + 'rotating_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_single_mmvae/visual/reconstructions_multimodal_pose.npy']

ideo_representations_filenames_vae = [      representations_root_path + 'rotating_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_ideo_mmvae/visual/reconstructions_multimodal_pose.npy']


full_representations_predictions = load_datasets(full_representations_filenames_vae, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames_vae, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames_vae, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon_vae = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon_vae = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon_vae = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon_vae = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon_vae = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon_vae = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

### ConvNet Reconstruction Quality ###

representations_root_path = 'C:/Users/Thomas/Downloads/HBP/representations/NRP/whiskeye_head_direction_'

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames_cnn = [      representations_root_path + 'rotating_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'circling_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_2_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_3_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_4_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_5_convnet_refined/visual/predictions_conv.npy']

single_representations_filenames_cnn = [    representations_root_path + 'rotating_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'circling_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_2_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_3_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_4_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_5_single_convnet_refined/visual/predictions_conv_single.npy']

ideo_representations_filenames_cnn = [      representations_root_path + 'rotating_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'circling_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_2_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_3_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_4_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_5_ideo_convnet_refined/visual/predictions_conv_ideo.npy']


full_representations_predictions = load_datasets(full_representations_filenames_cnn, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames_cnn, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames_cnn, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

### Jensen-Shannon ###

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon_cnn = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon_cnn = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon_cnn = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon_cnn = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon_cnn = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon_cnn = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

if mean_random is False:

    bar_width = 0.2

    ax1.bar([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8], full_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black')
    ax1.bar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], full_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black')
    ax1.bar([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], full_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black')
    ax1.set_xticks(range(1, 8))
    ax1.set_xticklabels(["Rot", "Circ", "R1", "R2", "R3", "R4", "R5"])

    ax2.bar([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8], single_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black')
    ax2.bar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], single_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black')
    ax2.bar([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], single_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black')
    ax2.set_xticks(range(1, 8))
    ax2.set_xticklabels(["Rot", "Circ", "R1", "R2", "R3", "R4", "R5"])

    ax3.bar([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8], ideo_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black')
    ax3.bar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], ideo_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black')
    ax3.bar([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2], ideo_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black')
    ax3.set_xticks(range(1, 8))
    ax3.set_xticklabels(["Rot", "Circ", "R1", "R2", "R3", "R4", "R5"])

if mean_random is True:

    bar_width = 0.5

    bar_outline_width = 0.5 # Different units to bar_width, thank God

    error_bar_width = 0.5

    full_jensen_shannon_pcn_standard_error = sem(full_jensen_shannon_pcn[2:5])
    single_jensen_shannon_pcn_standard_error = sem(single_jensen_shannon_pcn[2:5])
    ideo_jensen_shannon_pcn_standard_error = sem(ideo_jensen_shannon_pcn[2:5])

    full_jensen_shannon_pcn = [full_jensen_shannon_pcn[0], full_jensen_shannon_pcn[1], np.mean(full_jensen_shannon_pcn[2:5])]
    single_jensen_shannon_pcn = [single_jensen_shannon_pcn[0], single_jensen_shannon_pcn[1], np.mean(single_jensen_shannon_pcn[2:5])]
    ideo_jensen_shannon_pcn = [ideo_jensen_shannon_pcn[0], ideo_jensen_shannon_pcn[1], np.mean(ideo_jensen_shannon_pcn[2:5])]

    full_jensen_shannon_vae_standard_error = sem(full_jensen_shannon_vae[2:5])
    single_jensen_shannon_vae_standard_error = sem(single_jensen_shannon_vae[2:5])
    ideo_jensen_shannon_vae_standard_error = sem(ideo_jensen_shannon_vae[2:5])

    full_jensen_shannon_vae = [full_jensen_shannon_vae[0], full_jensen_shannon_vae[1], np.mean(full_jensen_shannon_vae[2:5])]
    single_jensen_shannon_vae = [single_jensen_shannon_vae[0], single_jensen_shannon_vae[1], np.mean(single_jensen_shannon_vae[2:5])]
    ideo_jensen_shannon_vae = [ideo_jensen_shannon_vae[0], ideo_jensen_shannon_vae[1], np.mean(ideo_jensen_shannon_vae[2:5])]

    full_jensen_shannon_cnn_standard_error = sem(full_jensen_shannon_cnn[2:5])
    single_jensen_shannon_cnn_standard_error = sem(single_jensen_shannon_cnn[2:5])
    ideo_jensen_shannon_cnn_standard_error = sem(ideo_jensen_shannon_cnn[2:5])

    full_jensen_shannon_cnn = [full_jensen_shannon_cnn[0], full_jensen_shannon_cnn[1], np.mean(full_jensen_shannon_cnn[2:5])]
    single_jensen_shannon_cnn = [single_jensen_shannon_cnn[0], single_jensen_shannon_cnn[1], np.mean(single_jensen_shannon_cnn[2:5])]
    ideo_jensen_shannon_cnn = [ideo_jensen_shannon_cnn[0], ideo_jensen_shannon_cnn[1], np.mean(ideo_jensen_shannon_cnn[2:5])]

    ax1.bar([1.5, 3.5, 5.5], full_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax1.bar([2.0, 4.0, 6.0], full_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black', linewidth = 0.5)
    ax1.bar([2.5, 4.5, 6.5], full_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax1.set_xticks(range(2, 8, 2))
    ax1.set_xticklabels(["Rotating", "Circling", "Random Walks"])

    ax1.errorbar(   x =     [5.5, 6.0, 6.5], 
                    y =     [full_jensen_shannon_pcn[2], full_jensen_shannon_vae[2], full_jensen_shannon_cnn[2]],
                    yerr =  [full_jensen_shannon_pcn_standard_error, full_jensen_shannon_vae_standard_error, full_jensen_shannon_cnn_standard_error],
                    fmt = 'none',
                    ecolor = 'Black',
                    capsize = 4.0,
                    elinewidth = 0.5,
                    mew = 0.5)

    ax2.bar([1.5, 3.5, 5.5], single_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax2.bar([2.0, 4.0, 6.0], single_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black', linewidth = 0.5)
    ax2.bar([2.5, 4.5, 6.5], single_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax2.set_xticks(range(2, 8, 2))
    ax2.set_xticklabels(["Rotating", "Circling", "Random Walks"])

    ax2.errorbar(   x =     [5.5, 6.0, 6.5], 
                    y =     [single_jensen_shannon_pcn[2], single_jensen_shannon_vae[2], single_jensen_shannon_cnn[2]],
                    yerr =  [single_jensen_shannon_pcn_standard_error, single_jensen_shannon_vae_standard_error, single_jensen_shannon_cnn_standard_error],
                    fmt = 'none',
                    ecolor = 'Black',
                    capsize = 4.0,
                    elinewidth = 0.5,
                    mew = 0.5)

    ax3.bar([1.5, 3.5, 5.5], ideo_jensen_shannon_pcn, width = bar_width, label = "PCN", color = pcn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax3.bar([2.0, 4.0, 6.0], ideo_jensen_shannon_vae, width = bar_width, label = "VAE", color = vae_colour, edgecolor = 'Black', linewidth = 0.5)
    ax3.bar([2.5, 4.5, 6.5], ideo_jensen_shannon_cnn, width = bar_width, label = "CNN", color = cnn_colour, edgecolor = 'Black', linewidth = 0.5)
    ax3.set_xticks(range(2, 8, 2))
    ax3.set_xticklabels(["Rotating", "Circling", "Random Walks"])

    ax3.errorbar(   x =     [5.5, 6.0, 6.5], 
                    y =     [ideo_jensen_shannon_pcn[2], ideo_jensen_shannon_vae[2], ideo_jensen_shannon_cnn[2]],
                    yerr =  [ideo_jensen_shannon_pcn_standard_error, ideo_jensen_shannon_vae_standard_error, ideo_jensen_shannon_cnn_standard_error],
                    fmt = 'none',
                    ecolor = 'Black',
                    capsize = 4.0,
                    elinewidth = 0.5,
                    mew = 0.5)

ax1.set_ylabel("Mean Sample RMSE")

ax1.set_title("Full Set")
ax2.set_title("Reduced Set")
ax3.set_title("SNN Estimate")

fig.tight_layout()

plt.subplots_adjust(wspace=0.1, hspace=0.2)

#plt.subplots_adjust(top = 0.2, bottom = 0.15)

plt.subplots_adjust(bottom = 0.2)

#fig.tight_layout()

#ax2.legend(bbox_to_anchor=(1.17, -0.1), ncol=3, prop={'size': 12})

#ax2.legend(bbox_to_anchor=(1.05, -0.1), ncol=3, prop={'size': 12})

#ax2.legend(bbox_to_anchor=(1.30, -0.1), ncol=3, prop={'size': 12})

#ax2.legend(bbox_to_anchor=(1.60, -0.1), ncol=3, prop={'size': 12})

ax2.legend(bbox_to_anchor=(1.10, -0.1), ncol=3, prop={'size': 12})

plt.show()
