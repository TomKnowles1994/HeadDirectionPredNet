import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, wasserstein_distance
from scipy.special import softmax, kl_div
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error

# 'ground_truth_to_prediction' means to take the error as being between the result and the ground truth for that data sample. It measures quality and accuracy
# 'prediction_to_prediction' means to take the error as being between the result and its ideal Laplacian form. It measures quality only

mode = 'ground_truth_to_prediction'

item = 1500

alpha = 1

def load_datasets(file_names, max_samples, clip = None, raise_floor = False, transpose = False):

    datasets = []

    for dataset in file_names:

        data = np.load(dataset)

        if transpose:

            data = data.T

        data = data[:max_samples]

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

datasets_root_path = ' ' # Point to datasets folder. Should be as written in Tensorflow code

representations_root_path = ' ' # Point to PredNet representations/predictions folder. Should be as written in Tensorflow code

results_root_path = ' ' # Point to results folder. Should be as written in Tensorflow code

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames = [      representations_root_path + 'rotating_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5/visual/reconstructions_head_direction.npy']

single_representations_filenames = [    representations_root_path + 'rotating_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5_single/visual/reconstructions_head_direction.npy']

ideo_representations_filenames = [      representations_root_path + 'rotating_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'circling_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_2_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_3_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_4_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'random_distal_5_ideo/visual/reconstructions_head_direction.npy']


full_representations_predictions = load_datasets(full_representations_filenames, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

fig, (( ax1, ax4, ax7),
      ( ax2, ax5, ax8),
      ( ax3, ax6, ax9)) = plt.subplots(3, 3, sharey = True)

fig.set_figheight(6.25)
fig.set_figwidth(11.25)

#fig.tight_layout()

### Jensen-Shannon ###

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

ax1.plot(ground_truth_datasets[0][item], label = "Ground Truth")
ax1.plot(full_representations_predictions[0][item], label = "Prediction")
ax2.plot(ground_truth_datasets[0][item])
ax2.plot(single_representations_predictions[0][item])
ax3.plot(ground_truth_datasets[0][item])
ax3.plot(ideo_representations_predictions[0][item])
x = np.arange(0.0, 180, 1)
# EPS doesn't support transparency, using lighter colour as a workaround with alpha = 1
ax1.fill_between(x, ground_truth_datasets[0][item], full_representations_predictions[0][item], color = '#99e6ff', label = "Difference", alpha = alpha)
ax2.fill_between(x, ground_truth_datasets[0][item], single_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)
ax3.fill_between(x, ground_truth_datasets[0][item], ideo_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)

### MMVAE Reconstruction Quality ###

representations_root_path = ' ' # Point to VAE representations/predictions folder. Should be as written in Tensorflow code

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames = [      representations_root_path + 'rotating_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_full_mmvae/visual/reconstructions_multimodal_pose.npy']

single_representations_filenames = [    representations_root_path + 'rotating_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_single_mmvae/visual/reconstructions_multimodal_pose.npy']

ideo_representations_filenames = [      representations_root_path + 'rotating_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'circling_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_2_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_3_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_4_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'random_distal_5_ideo_mmvae/visual/reconstructions_multimodal_pose.npy']


full_representations_predictions = load_datasets(full_representations_filenames, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

ax4.plot(ground_truth_datasets[0][item])
ax4.plot(full_representations_predictions[0][item])
ax5.plot(ground_truth_datasets[0][item])
ax5.plot(single_representations_predictions[0][item])
ax6.plot(ground_truth_datasets[0][item])
ax6.plot(ideo_representations_predictions[0][item])
x = np.arange(0.0, 180, 1)
ax4.fill_between(x, ground_truth_datasets[0][item], full_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)
ax5.fill_between(x, ground_truth_datasets[0][item], single_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)
ax6.fill_between(x, ground_truth_datasets[0][item], ideo_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)

### ConvNet Reconstruction Quality ###

representations_root_path = ' ' # Point to CNN representations/predictions folder. Should be as written in Tensorflow code

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'circling_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_2/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_3/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_4/networkOutput_gaussianised.npy',
                                        datasets_root_path + 'random_distal_5/networkOutput_gaussianised.npy']

full_representations_filenames = [      representations_root_path + 'rotating_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'circling_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_2_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_3_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_4_convnet_refined/visual/predictions_conv.npy',
                                        representations_root_path + 'random_distal_5_convnet_refined/visual/predictions_conv.npy']

single_representations_filenames = [    representations_root_path + 'rotating_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'circling_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_2_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_3_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_4_single_convnet_refined/visual/predictions_conv_single.npy',
                                        representations_root_path + 'random_distal_5_single_convnet_refined/visual/predictions_conv_single.npy']

ideo_representations_filenames = [      representations_root_path + 'rotating_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'circling_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_2_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_3_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_4_ideo_convnet_refined/visual/predictions_conv_ideo.npy',
                                        representations_root_path + 'random_distal_5_ideo_convnet_refined/visual/predictions_conv_ideo.npy']


full_representations_predictions = load_datasets(full_representations_filenames, 3000, clip = None, raise_floor = False)

single_representations_predictions = load_datasets(single_representations_filenames, 3000, clip = None, raise_floor = False)

ideo_representations_predictions = load_datasets(ideo_representations_filenames, 3000, clip = None, raise_floor = False)

full_representations_laplacians = convert_to_laplacians(full_representations_predictions)

single_representations_laplacians = convert_to_laplacians(single_representations_predictions)

ideo_representations_laplacians = convert_to_laplacians(ideo_representations_predictions)

### Jensen-Shannon ###

if mode is 'prediction_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = True)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, full_representations_laplacians, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, single_representations_laplacians, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ideo_representations_laplacians, 'rmse')

if mode is 'ground_truth_to_prediction':

    ground_truth_datasets = load_datasets(ground_truth_filenames, 3000, clip = None, raise_floor = False, transpose = False)

    full_jensen_shannon = calculate_divergence(full_representations_predictions, ground_truth_datasets, 'rmse')

    single_jensen_shannon = calculate_divergence(single_representations_predictions, ground_truth_datasets, 'rmse')

    ideo_jensen_shannon = calculate_divergence(ideo_representations_predictions, ground_truth_datasets, 'rmse')

ax7.plot(ground_truth_datasets[0][item])
ax7.plot(full_representations_predictions[0][item])
ax8.plot(ground_truth_datasets[0][item])
ax8.plot(single_representations_predictions[0][item])
ax9.plot(ground_truth_datasets[0][item])
ax9.plot(ideo_representations_predictions[0][item])
x = np.arange(0.0, 180, 1)
ax7.fill_between(x, ground_truth_datasets[0][item], full_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)
ax8.fill_between(x, ground_truth_datasets[0][item], single_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)
ax9.fill_between(x, ground_truth_datasets[0][item], ideo_representations_predictions[0][item], color = '#99e6ff', alpha = alpha)

ax1.set_ylabel("Full Set")
ax2.set_ylabel("Reduced Set")
ax3.set_ylabel("SNN Estimate")

ax3.set_xlabel("Prediction Index")
ax6.set_xlabel("Prediction Index")
ax9.set_xlabel("Prediction Index")

ax1.set_title("PCN")
ax4.set_title("VAE")
ax7.set_title("CNN")

plt.subplots_adjust(wspace=0.1, hspace=0.25)

plt.subplots_adjust(bottom = 0.2)

ax1.legend(bbox_to_anchor=(2.58, -2.95), ncol=3, prop={'size': 12})

plt.show()
