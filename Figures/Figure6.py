import numpy as np
import pandas as pd
from scipy.stats import norm, laplace, wasserstein_distance
from scipy.special import softmax, kl_div
from scipy.spatial.distance import jensenshannon
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
from cv2 import imread
from sklearn.metrics import mean_squared_error

plot_type = 'by_training_set'

mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12)

pcn_colour = '#63F500'#BD25E6'#F24E30'#F53313'#F56700'#63F500'
vae_colour = '#09D4F5'#F74F31'#BD25E6'#39E3C7'#3A78CF'#09D4F5'
cnn_colour = '#F70C7D'#5BE854'#8925BD'#F70C7D'

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

def laplacianise(prediction):

    sharpness = 20

    N = 180

    gaussian_width = prediction.shape[1]

    gaussian_range = np.arange(-(gaussian_width//2),(gaussian_width//2)) # Used to be up to (gaussian_width//2)+1 (adjusts for odd numbers)

    gaussian_block = np.resize(gaussian_range, new_shape = (prediction.shape))

    max_locations = np.argmax(prediction, axis = 1)

    pose_gaussians = laplace(0, gaussian_width//sharpness)

    zeroed_gaussians = pose_gaussians.pdf(gaussian_block)

    shifted_gaussians = np.empty_like(prediction)

    shifted = np.roll(zeroed_gaussians[0], max_locations[0]-(N//2)) # Used to be hard-coded to -31 i.e. -63//2

    for index in range(len(max_locations)):
        shifted = np.roll(zeroed_gaussians[index,:], max_locations[index]-(N//2))
        shifted_gaussians[index,:] = shifted

    return shifted_gaussians

def calculate_divergence(prediction_list, laplacian_list, metric):

    total_distances_list = []

    if metric is 'wasserstein':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            wasserstein_distances = []

            for prediction, laplacian in zip(predictions, laplacians):

                wasserstein_distances.append(wasserstein_distance(prediction, laplacian))

            total_distances_list.append(np.mean(wasserstein_distances))

    elif metric is 'kl':

        for predictions, laplacians in zip(prediction_list, laplacian_list):

            kl_divergences = []

            for prediction, laplacian in zip(predictions, laplacians):

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

def append_degree_symbol_to_tick_labels(tick_labels):

    degree_symbol = u'\N{DEGREE SIGN}'

    tick_labels = [str(x) + degree_symbol for x in tick_labels]

    return tick_labels

datasets_root_path = ' ' # Point to folder containing dataset folders

representations_root_path = ' ' # Point to folder of representation/prediction folders

results_root_path = ' ' # Point to folder showing results

ground_truth_filenames = [              datasets_root_path + 'rotating_distal/networkOutput_gaussianised.npy']

full_representations_filenames = [      representations_root_path + 'rotating_distal/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'rotating_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'rotating_distal_full_convnet_refined/visual/predictions_conv.npy']

single_representations_filenames = [    representations_root_path + 'rotating_distal_single/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'rotating_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'rotating_distal_single_convnet_refined/visual/predictions_conv_single.npy']

ideo_representations_filenames = [      representations_root_path + 'rotating_distal_ideo/visual/reconstructions_head_direction.npy',
                                        representations_root_path + 'rotating_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy',
                                        representations_root_path + 'rotating_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy']

multiprednet_representations_filenames = [      representations_root_path + 'rotating_distal/visual/reconstructions_head_direction.npy',
                                                representations_root_path + 'rotating_distal_single/visual/reconstructions_head_direction.npy',
                                                representations_root_path + 'rotating_distal_ideo/visual/reconstructions_head_direction.npy']

mmvae_representations_filenames = [             representations_root_path + 'rotating_distal_full_mmvae/visual/reconstructions_multimodal_pose.npy',
                                                representations_root_path + 'rotating_distal_single_mmvae/visual/reconstructions_multimodal_pose.npy',
                                                representations_root_path + 'rotating_distal_ideo_mmvae/visual/reconstructions_multimodal_pose.npy']

convnet_representations_filenames = [           representations_root_path + 'rotating_distal_convnet_refined/visual/predictions_conv.npy',
                                                representations_root_path + 'rotating_distal_single_convnet_refined/visual/predictions_conv_single.npy',
                                                representations_root_path + 'rotating_distal_ideo_convnet_refined/visual/predictions_conv_ideo.npy']

def generate_radar_chart(   predictions_filepath, ax, colour_string, label, flip = False, mean = False, error_bars = False, 
                            ground_truth_filepath = None, raise_floor = False, bar = False, bar_label = False):

    if mean == True:

        datasets = []

        for filename in predictions_filepath:

            data = np.load(filename)[:3000]

            datasets.append(data)

        predictions = np.mean(datasets, axis = 0)

        assert predictions.shape == data.shape

    else:

        predictions = np.load(predictions_filepath)[:3000]

        if raise_floor == True:

            predictions = predictions + np.abs(np.min(predictions)) + 0.0001

    if np.sum(np.isnan(predictions)) > 0:

        print("Warning: NaNs in dataset {}".format(predictions_filepath))

    predictions[np.where(predictions < 0.001)] = 0.001

    if ground_truth_filepath is None:

        predictions0_29 = predictions[(np.argmax(predictions, axis = 1) >= 0) & (np.argmax(predictions, axis = 1) <= 15)]
        predictions30_59 = predictions[(np.argmax(predictions, axis = 1) >= 16) & (np.argmax(predictions, axis = 1) <= 30)]
        predictions60_89 = predictions[(np.argmax(predictions, axis = 1) >= 31) & (np.argmax(predictions, axis = 1) <= 45)]
        predictions90_119 = predictions[(np.argmax(predictions, axis = 1) >= 46) & (np.argmax(predictions, axis = 1) <= 60)]
        predictions120_149 = predictions[(np.argmax(predictions, axis = 1) >= 61) & (np.argmax(predictions, axis = 1) <= 75)]
        predictions150_179 = predictions[(np.argmax(predictions, axis = 1) >= 76) & (np.argmax(predictions, axis = 1) <= 90)]
        predictions180_209 = predictions[(np.argmax(predictions, axis = 1) >= 91) & (np.argmax(predictions, axis = 1) <= 105)]
        predictions210_239 = predictions[(np.argmax(predictions, axis = 1) >= 106) & (np.argmax(predictions, axis = 1) <= 120)]
        predictions240_269 = predictions[(np.argmax(predictions, axis = 1) >= 121) & (np.argmax(predictions, axis = 1) <= 135)]
        predictions270_299 = predictions[(np.argmax(predictions, axis = 1) >= 136) & (np.argmax(predictions, axis = 1) <= 150)]
        predictions300_329 = predictions[(np.argmax(predictions, axis = 1) >= 151) & (np.argmax(predictions, axis = 1) <= 165)]
        predictions330_359 = predictions[(np.argmax(predictions, axis = 1) >= 166) & (np.argmax(predictions, axis = 1) <= 180)]

        laplacianised_predictions0_29 = laplacianise(predictions0_29)
        laplacianised_predictions30_59 = laplacianise(predictions30_59)
        laplacianised_predictions60_89 = laplacianise(predictions60_89)
        laplacianised_predictions90_119 = laplacianise(predictions90_119)
        laplacianised_predictions120_149 = laplacianise(predictions120_149)
        laplacianised_predictions150_179 = laplacianise(predictions150_179)
        laplacianised_predictions180_209 = laplacianise(predictions180_209)
        laplacianised_predictions210_239 = laplacianise(predictions210_239)
        laplacianised_predictions240_269 = laplacianise(predictions240_269)
        laplacianised_predictions270_299 = laplacianise(predictions270_299)
        laplacianised_predictions300_329 = laplacianise(predictions300_329)
        laplacianised_predictions330_359 = laplacianise(predictions330_359)

        jensen_shannon0_29 = mean_squared_error(predictions0_29, laplacianised_predictions0_29, squared = False)
        jensen_shannon30_59 = mean_squared_error(predictions30_59, laplacianised_predictions30_59, squared = False)
        jensen_shannon60_89 = mean_squared_error(predictions60_89, laplacianised_predictions60_89, squared = False)
        jensen_shannon90_119 = mean_squared_error(predictions90_119, laplacianised_predictions90_119, squared = False)
        jensen_shannon120_149 = mean_squared_error(predictions120_149, laplacianised_predictions120_149, squared = False)
        jensen_shannon150_179 = mean_squared_error(predictions150_179, laplacianised_predictions150_179, squared = False)
        jensen_shannon180_209 = mean_squared_error(predictions180_209, laplacianised_predictions180_209, squared = False)
        jensen_shannon210_239 = mean_squared_error(predictions210_239, laplacianised_predictions210_239, squared = False)
        jensen_shannon240_269 = mean_squared_error(predictions240_269, laplacianised_predictions240_269, squared = False)
        jensen_shannon270_299 = mean_squared_error(predictions270_299, laplacianised_predictions270_299, squared = False)
        jensen_shannon300_329 = mean_squared_error(predictions300_329, laplacianised_predictions300_329, squared = False)
        jensen_shannon330_359 = mean_squared_error(predictions330_359, laplacianised_predictions330_359, squared = False)

    else:

        datasets = []

        if mean == True:

            datasets = []

            for filename in ground_truth_filepath:

                data = np.load(filename)[:3000]

                datasets.append(data)

            ground_truth = np.mean(datasets, axis = 0)

            assert predictions.shape == data.shape

        else:

            ground_truth = np.load(ground_truth_filepath)[:3000]

        ground_truth[np.where(ground_truth < 0.001)] = 0.001

        ground_truth0_29_indices = np.where((np.argmax(ground_truth, axis = 1) >= 0) & (np.argmax(ground_truth, axis = 1) <= 15))
        ground_truth30_59_indices = np.where((np.argmax(ground_truth, axis = 1) >= 16) & (np.argmax(ground_truth, axis = 1) <= 30))
        ground_truth60_89_indices = np.where((np.argmax(ground_truth, axis = 1) >= 31) & (np.argmax(ground_truth, axis = 1) <= 45))
        ground_truth90_119_indices = np.where((np.argmax(ground_truth, axis = 1) >= 46) & (np.argmax(ground_truth, axis = 1) <= 60))
        ground_truth120_149_indices = np.where((np.argmax(ground_truth, axis = 1) >= 61) & (np.argmax(ground_truth, axis = 1) <= 75))
        ground_truth150_179_indices = np.where((np.argmax(ground_truth, axis = 1) >= 76) & (np.argmax(ground_truth, axis = 1) <= 90))
        ground_truth180_209_indices = np.where((np.argmax(ground_truth, axis = 1) >= 91) & (np.argmax(ground_truth, axis = 1) <= 105))
        ground_truth210_239_indices = np.where((np.argmax(ground_truth, axis = 1) >= 106) & (np.argmax(ground_truth, axis = 1) <= 120))
        ground_truth240_269_indices = np.where((np.argmax(ground_truth, axis = 1) >= 121) & (np.argmax(ground_truth, axis = 1) <= 135))
        ground_truth270_299_indices = np.where((np.argmax(ground_truth, axis = 1) >= 136) & (np.argmax(ground_truth, axis = 1) <= 150))
        ground_truth300_329_indices = np.where((np.argmax(ground_truth, axis = 1) >= 151) & (np.argmax(ground_truth, axis = 1) <= 165))
        ground_truth330_359_indices = np.where((np.argmax(ground_truth, axis = 1) >= 166) & (np.argmax(ground_truth, axis = 1) <= 180))

        ground_truth0_29 = ground_truth[ground_truth0_29_indices]
        ground_truth30_59 = ground_truth[ground_truth30_59_indices]
        ground_truth60_89 = ground_truth[ground_truth60_89_indices]
        ground_truth90_119 = ground_truth[ground_truth90_119_indices]
        ground_truth120_149 = ground_truth[ground_truth120_149_indices]
        ground_truth150_179 = ground_truth[ground_truth150_179_indices]
        ground_truth180_209 = ground_truth[ground_truth180_209_indices]
        ground_truth210_239 = ground_truth[ground_truth210_239_indices]
        ground_truth240_269 = ground_truth[ground_truth240_269_indices]
        ground_truth270_299 = ground_truth[ground_truth270_299_indices]
        ground_truth300_329 = ground_truth[ground_truth300_329_indices]
        ground_truth330_359 = ground_truth[ground_truth330_359_indices]

        predictions0_29 = predictions[ground_truth0_29_indices]
        predictions30_59 = predictions[ground_truth30_59_indices]
        predictions60_89 = predictions[ground_truth60_89_indices]
        predictions90_119 = predictions[ground_truth90_119_indices]
        predictions120_149 = predictions[ground_truth120_149_indices]
        predictions150_179 = predictions[ground_truth150_179_indices]
        predictions180_209 = predictions[ground_truth180_209_indices]
        predictions210_239 = predictions[ground_truth210_239_indices]
        predictions240_269 = predictions[ground_truth240_269_indices]
        predictions270_299 = predictions[ground_truth270_299_indices]
        predictions300_329 = predictions[ground_truth300_329_indices]
        predictions330_359 = predictions[ground_truth330_359_indices]

        laplacianised_predictions0_29 = laplacianise(predictions0_29)
        laplacianised_predictions30_59 = laplacianise(predictions30_59)
        laplacianised_predictions60_89 = laplacianise(predictions60_89)
        laplacianised_predictions90_119 = laplacianise(predictions90_119)
        laplacianised_predictions120_149 = laplacianise(predictions120_149)
        laplacianised_predictions150_179 = laplacianise(predictions150_179)
        laplacianised_predictions180_209 = laplacianise(predictions180_209)
        laplacianised_predictions210_239 = laplacianise(predictions210_239)
        laplacianised_predictions240_269 = laplacianise(predictions240_269)
        laplacianised_predictions270_299 = laplacianise(predictions270_299)
        laplacianised_predictions300_329 = laplacianise(predictions300_329)
        laplacianised_predictions330_359 = laplacianise(predictions330_359)

        jensen_shannon0_29 = mean_squared_error(ground_truth0_29, predictions0_29, squared = False)
        jensen_shannon30_59 = mean_squared_error(ground_truth30_59, predictions30_59, squared = False)
        jensen_shannon60_89 = mean_squared_error(ground_truth60_89, predictions60_89, squared = False)
        jensen_shannon90_119 = mean_squared_error(ground_truth90_119, predictions90_119, squared = False)
        jensen_shannon120_149 = mean_squared_error(ground_truth120_149, predictions120_149, squared = False)
        jensen_shannon150_179 = mean_squared_error(ground_truth150_179, predictions150_179, squared = False)
        jensen_shannon180_209 = mean_squared_error(ground_truth180_209, predictions180_209, squared = False)
        jensen_shannon210_239 = mean_squared_error(ground_truth210_239, predictions210_239, squared = False)
        jensen_shannon240_269 = mean_squared_error(ground_truth240_269, predictions240_269, squared = False)
        jensen_shannon270_299 = mean_squared_error(ground_truth270_299, predictions270_299, squared = False)
        jensen_shannon300_329 = mean_squared_error(ground_truth300_329, predictions300_329, squared = False)
        jensen_shannon330_359 = mean_squared_error(ground_truth330_359, predictions330_359, squared = False)

    if flip:

        line_heights = [np.mean(jensen_shannon180_209), np.mean(jensen_shannon210_239), np.mean(jensen_shannon240_269),
                    np.mean(jensen_shannon270_299), np.mean(jensen_shannon300_329), np.mean(jensen_shannon330_359),
                    np.mean(jensen_shannon0_29), np.mean(jensen_shannon30_59), np.mean(jensen_shannon60_89),
                    np.mean(jensen_shannon90_119), np.mean(jensen_shannon120_149), np.mean(jensen_shannon150_179),
                    np.mean(jensen_shannon180_209)]

    else:

        line_heights = [np.mean(jensen_shannon0_29), np.mean(jensen_shannon30_59), np.mean(jensen_shannon60_89),
                    np.mean(jensen_shannon90_119), np.mean(jensen_shannon120_149), np.mean(jensen_shannon150_179),
                    np.mean(jensen_shannon180_209), np.mean(jensen_shannon210_239), np.mean(jensen_shannon240_269),
                    np.mean(jensen_shannon270_299), np.mean(jensen_shannon300_329), np.mean(jensen_shannon330_359),
                    np.mean(jensen_shannon0_29)]

    angles = np.radians([0,30,60,90,120,150,180,210,240,270,300,330,0]) + np.radians(15)

    if bar:

        segment_heights = [ ground_truth0_29.shape[0], ground_truth30_59.shape[0], ground_truth60_89.shape[0], ground_truth90_119.shape[0],
                        ground_truth120_149.shape[0], ground_truth150_179.shape[0], ground_truth180_209.shape[0], ground_truth210_239.shape[0],
                        ground_truth240_269.shape[0], ground_truth270_299.shape[0], ground_truth300_329.shape[0], ground_truth330_359.shape[0],
                        ground_truth0_29.shape[0]]

        cmap = mpl.cm.get_cmap('Reds')

        norm = mpl.colors.Normalize(vmin=np.min(segment_heights)/1.2, vmax=np.max(segment_heights))
        
        COLOURS = cmap(norm(segment_heights))

        if bar_label:

            bar = ax.bar(angles, segment_heights, width = 0.4, color = COLOURS, zorder = 10, label = 'Samples')

            print("yes")

        else:

            bar = ax.bar(angles, segment_heights, width = 0.4, color = COLOURS, zorder = 10)

            print("no")

        scaling_factor = np.mean(segment_heights)/np.mean(line_heights)

        line_heights = [x * scaling_factor for x in line_heights]

    ax.set_xticks(np.radians([0,30,60,90,120,150,180,210,240,270,300,330]))

    ax.set_rlabel_position(0)

    if bar:

        ax.tick_params(labelleft=False, labelright=False,
                        labeltop=False, labelbottom=True)

        ax.set_rorigin(-100)

    else:

        ax.tick_params(labelleft=True, labelright=False,
                        labeltop=False, labelbottom=True)

        ax.set_rorigin(-0.1)

    ax.set_theta_offset((np.pi / 2))

    ax.plot(angles, line_heights, 'o-', color = colour_string, zorder = 10, label = label)

    if error_bars == True:

        error = [np.std(jensen_shannon0_29), np.std(jensen_shannon30_59), np.std(jensen_shannon60_89),
                        np.std(jensen_shannon90_119), np.std(jensen_shannon120_149), np.std(jensen_shannon150_179),
                        np.std(jensen_shannon180_209), np.std(jensen_shannon210_239), np.std(jensen_shannon240_269),
                        np.std(jensen_shannon270_299), np.std(jensen_shannon300_329), np.std(jensen_shannon330_359),
                        np.std(jensen_shannon0_29)]

        error = [1/x for x in error]
        error = [x/10 for x in error]

        ax.errorbar(angles, line_heights, yerr=error, xerr=0, capsize=0, color = colour_string, fmt = 'none')

    return ax

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = True, subplot_kw={"projection": "polar"})

fig.set_figheight(9)
fig.set_figwidth(16)

colourmap = 'Oranges'

# old yellow colour was #F5DE22

if plot_type is 'by_model':

    ax1 = generate_radar_chart(full_representations_filenames[0], ax1, pcn_colour, label = 'Full Set', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax1 = generate_radar_chart(single_representations_filenames[0], ax1, vae_colour, label = 'Single Rotation', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax1 = generate_radar_chart(ideo_representations_filenames[0], ax1, cnn_colour, label = 'Ideothetic Estimate', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)

    ax1.set_title("MultiPredNet", pad = 20)

    ax2 = generate_radar_chart(full_representations_filenames[1], ax2, pcn_colour, label = 'Full Set', flip = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax2 = generate_radar_chart(single_representations_filenames[1], ax2, vae_colour, label = 'Single Rotation', flip = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax2 = generate_radar_chart(ideo_representations_filenames[1], ax2, cnn_colour, label = 'Ideothetic Estimate', flip = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)

    ax2.set_title("MMVAE", pad = 20)
    ax2.legend(bbox_to_anchor=(1.25, -0.12), ncol=3, prop={'size': 12})

    ax3 = generate_radar_chart(full_representations_filenames[2], ax3, pcn_colour, label = 'Full Set', error_bars = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax3 = generate_radar_chart(single_representations_filenames[2], ax3, vae_colour, label = 'Single Rotation', error_bars = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax3 = generate_radar_chart(ideo_representations_filenames[2], ax3, cnn_colour, label = 'Ideothetic Estimate', error_bars = False, ground_truth_filepath = ground_truth_filenames[0], raise_floor = False, bar_label = True)

    ax3.set_title("CNN", pad = 20)

if plot_type is 'by_training_set':

    ax1 = generate_radar_chart(multiprednet_representations_filenames[0], ax1, pcn_colour, label = 'PCN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax1 = generate_radar_chart(mmvae_representations_filenames[0], ax1, vae_colour, label = 'VAE', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax1 = generate_radar_chart(convnet_representations_filenames[0], ax1, cnn_colour, label = 'CNN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)

    ax1.set_title("Full Set", pad = 20)

    ax2 = generate_radar_chart(multiprednet_representations_filenames[1], ax2, pcn_colour, label = 'PCN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax2 = generate_radar_chart(mmvae_representations_filenames[1], ax2, vae_colour, label = 'VAE', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax2 = generate_radar_chart(convnet_representations_filenames[1], ax2, cnn_colour, label = 'CNN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)

    ax2.set_title("Reduced Set", pad = 20)
    ax2.legend(bbox_to_anchor=(0.97, -0.12), ncol=3, prop={'size': 12})

    ax3 = generate_radar_chart(multiprednet_representations_filenames[2], ax3, pcn_colour, label = 'PCN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax3 = generate_radar_chart(mmvae_representations_filenames[2], ax3, vae_colour, label = 'VAE', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)
    ax3 = generate_radar_chart(convnet_representations_filenames[2], ax3, cnn_colour, label = 'CNN', ground_truth_filepath = ground_truth_filenames[0], raise_floor = False)

    ax3.set_title("SNN Estimate", pad = 20)

plt.subplots_adjust(top = 1.25, bottom = 0.05, wspace = 0.22)

ax4 = fig.add_subplot(4,1,4)

whiskeye_vision = imread(' Path to image file ')[:,:,::-1]

ax4.imshow(whiskeye_vision)

degree_symbol = u'\N{DEGREE SIGN}'

image_labels = [180, 210, 240, 270, 300, 330, 0, 30, 60, 90, 120, 150, 180]

image_labels = append_degree_symbol_to_tick_labels(image_labels)

tick_locations = np.linspace(0, whiskeye_vision.shape[1], num = len(image_labels))

ax4.set_xticks([])

ax4.set_xticks(tick_locations)

ax4.set_xticklabels(image_labels)

ax4.set_yticks([])

plt.show()

def generate_radial_bar_chart(ground_truth_filepath, ax, colourmap_string = None):

    ground_truth = np.load(ground_truth_filepath)[:3000]

    ground_truth[np.where(ground_truth < 0.001)] = 0.001

    ground_truth0_29_indices = np.where((np.argmax(ground_truth, axis = 1) >= 0) & (np.argmax(ground_truth, axis = 1) <= 15))
    ground_truth30_59_indices = np.where((np.argmax(ground_truth, axis = 1) >= 16) & (np.argmax(ground_truth, axis = 1) <= 30))
    ground_truth60_89_indices = np.where((np.argmax(ground_truth, axis = 1) >= 31) & (np.argmax(ground_truth, axis = 1) <= 45))
    ground_truth90_119_indices = np.where((np.argmax(ground_truth, axis = 1) >= 46) & (np.argmax(ground_truth, axis = 1) <= 60))
    ground_truth120_149_indices = np.where((np.argmax(ground_truth, axis = 1) >= 61) & (np.argmax(ground_truth, axis = 1) <= 75))
    ground_truth150_179_indices = np.where((np.argmax(ground_truth, axis = 1) >= 76) & (np.argmax(ground_truth, axis = 1) <= 90))
    ground_truth180_209_indices = np.where((np.argmax(ground_truth, axis = 1) >= 91) & (np.argmax(ground_truth, axis = 1) <= 105))
    ground_truth210_239_indices = np.where((np.argmax(ground_truth, axis = 1) >= 106) & (np.argmax(ground_truth, axis = 1) <= 120))
    ground_truth240_269_indices = np.where((np.argmax(ground_truth, axis = 1) >= 121) & (np.argmax(ground_truth, axis = 1) <= 135))
    ground_truth270_299_indices = np.where((np.argmax(ground_truth, axis = 1) >= 136) & (np.argmax(ground_truth, axis = 1) <= 150))
    ground_truth300_329_indices = np.where((np.argmax(ground_truth, axis = 1) >= 151) & (np.argmax(ground_truth, axis = 1) <= 165))
    ground_truth330_359_indices = np.where((np.argmax(ground_truth, axis = 1) >= 166) & (np.argmax(ground_truth, axis = 1) <= 180))

    ground_truth0_29 = ground_truth[ground_truth0_29_indices]
    ground_truth30_59 = ground_truth[ground_truth30_59_indices]
    ground_truth60_89 = ground_truth[ground_truth60_89_indices]
    ground_truth90_119 = ground_truth[ground_truth90_119_indices]
    ground_truth120_149 = ground_truth[ground_truth120_149_indices]
    ground_truth150_179 = ground_truth[ground_truth150_179_indices]
    ground_truth180_209 = ground_truth[ground_truth180_209_indices]
    ground_truth210_239 = ground_truth[ground_truth210_239_indices]
    ground_truth240_269 = ground_truth[ground_truth240_269_indices]
    ground_truth270_299 = ground_truth[ground_truth270_299_indices]
    ground_truth300_329 = ground_truth[ground_truth300_329_indices]
    ground_truth330_359 = ground_truth[ground_truth330_359_indices]

    segment_heights = [ ground_truth0_29.shape[0], ground_truth30_59.shape[0], ground_truth60_89.shape[0], ground_truth90_119.shape[0],
                        ground_truth120_149.shape[0], ground_truth150_179.shape[0], ground_truth180_209.shape[0], ground_truth210_239.shape[0],
                        ground_truth240_269.shape[0], ground_truth270_299.shape[0], ground_truth300_329.shape[0], ground_truth330_359.shape[0]]

    angles = np.radians([0,30,60,90,120,150,180,210,240,270,300,330]) + np.radians(15)

    ax.set_xticks(np.radians([0,30,60,90,120,150,180,210,240,270,300,330]))

    ax.set_rlabel_position(30)

    ax.tick_params(labelleft=False, labelright=False,
                labeltop=False, labelbottom=True)

    ax.set_rorigin(-200)

    ax.set_theta_offset((np.pi / 2))

    norm = mpl.colors.Normalize(vmin=np.min(segment_heights)/1.2, vmax=np.max(segment_heights))

    if colourmap_string is None:

        COLOURS = ["#6C5B7B","#C06C84","#F67280","#F8B195"]

        norm = mpl.colors.Normalize(vmin=np.min(segment_heights)/1.2, vmax=np.max(segment_heights))

    else:

        cmap = mpl.cm.get_cmap(colourmap_string)

    COLOURS = cmap(norm(segment_heights))

    bar = ax.bar(angles, segment_heights, width = 0.4, color = COLOURS, zorder = 10)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

    sm.set_array([])

    return sm, ax

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={"projection": "polar"})

fig.set_figheight(8)
fig.set_figwidth(17)

colourmap = 'Blues'

sm1, ax1 = generate_radial_bar_chart(ground_truth_filenames[0], ax1, colourmap)
sm2, ax2 = generate_radial_bar_chart(ground_truth_filenames[0], ax2, colourmap)
sm3, ax3 = generate_radial_bar_chart(ground_truth_filenames[0], ax3, colourmap)

ax1.set_title("MultiPredNet", pad = 20)
ax2.set_title("Single Rotation", pad = 20)
ax3.set_title("Ideothetic Estimate", pad = 20)

cb = fig.colorbar(sm3, ax=[[ax1, ax2, ax3]], orientation = 'horizontal', pad = 0.1, fraction = 0.15)

cb.set_label('Number of Samples')

fig.suptitle("MultiPredNet Trained on Random #2 Dataset")#, y = 0.85)

plt.show()
