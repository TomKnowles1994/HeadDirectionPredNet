import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Seed value
seed_value = 29384767

# 1. Set PYTHONHASHSEED environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set Python built-in pseudo-random generator at a fixed value
random.seed(seed_value)

# 3. Set numpy pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the Tensorflow pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)

### Data Processing Functions

def load_npy_data(data_path, sample_count, minibatch_size, shuffle=True):

    images_file = '/images.npy'

    pose_file = '/networkOutput_gaussianised.npy'

    img = np.load(data_path + images_file)

    if img.shape[0] > sample_count:

        img = img[:sample_count]

    img = img.reshape(img.shape[0], 10800) # flatten

    pose_data = np.load(data_path + pose_file)
    
    # Sanity checks to ensure data fits within minibatch size; trim if not

    if pose_data.shape[0] > sample_count:

        pose_data = pose_data[:sample_count]

    if img.shape[0] is not pose_data.shape[0]:

        min_sample_count = min(img.shape[0], pose_data.shape[0])

        img = img[:min_sample_count]
        print(img.shape)

        pose_data = pose_data[:min_sample_count]
        print(pose_data.shape)

    if img.shape[0] % minibatch_size is not 0:

        new_shape = img.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        img = img[:new_shape]

    if pose_data.shape[0] % minibatch_size is not 0:

        new_shape = pose_data.shape[0]//minibatch_size
        new_shape = new_shape * minibatch_size

        pose_data = pose_data[:new_shape]

    if shuffle:
        # shuffle sequence of data but maintain visual-pose alignment
        img, pose_data = shuffle_in_sync(img, pose_data)

    print("Final image data shape: {}".format(img.shape))
    print("Final HD data shape: {}".format(pose_data.shape))

    return img, pose_data

def shuffle_in_sync(visual_data, pose_data):

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_pose = visual_data[shared_indices], pose_data[shared_indices]

    return shuffled_visual, shuffled_pose

def circular_mean(causes, gaussian = False):

    causes_length = causes.get_shape().as_list()[1]

    arc_length = (2*np.pi)/causes_length

    angles_in_radians = np.arange(causes_length) * arc_length

    cos_mean = np.cos(angles_in_radians)

    sin_mean = np.sin(angles_in_radians)

    circular_mean = np.arctan2(sin_mean, cos_mean)

    if gaussian is True:

        gaussian_range = np.arange(-(causes_length//2),(causes_length//2))

        gaussian_function = norm(circular_mean, causes_length//10)

        gaussian_mean = gaussian_function.pdf(gaussian_range)

        circular_mean = minmax_scale(gaussian_mean)

    return circular_mean

### User-defined Parameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

n_sample = 2000                             # If you have collected your own dataset, you will need to determine how many samples where collected in the run
                                            # Alternatively, if you are using a built-in dataset, copy the sample number as described in the datasets' README

minibatch_sz = 10                            # Minibatch size. Can be left as default for physical data, for simulated data good numbers to try are 40, 50 and 100
                                            # Datasize size must be fully divisible by minibatch size

data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_trainingset'               # Path to training data. Training data should be in .npy format:

save_path = 'C:/Users/Thomas/Downloads/HBP/model_checkpoints/landmarks_vh/whiskeye_head_direction_ideo'#trial1'               # Path to save trained model to (once trained)
load_path = 'C:/Users/Thomas/Downloads/HBP/model_checkpoints/landmarks_vh/whiskeye_head_direction_ideo'#trial1'               # Path to load trained model from (after training, or if already trained beforehand)

cause_save_path = save_path + '/causes'#trial1/causes'  # Path to save causes to (optional, for training diagnostics)
reconstruction_save_path = save_path + '/reconstructions'#trial1/reconstructions'  # Path to save reconstructons to (optional, for training diagnostics)

save_m1_causes = False
save_m2_causes = False
save_msi_causes = False

save_m1_reconstructions = False
save_msi_reconstructions = True

n_epoch = 200                               # Number of training epochs to generate model. Default is 200
                                            
shuffle_data = False                        # Do you want to shuffle the training data? Default is False

# Load the data from .mat files

#visual_data, pose_data = load_mat_data(data_path, shuffle_data)

# Alternatively, load the data from .npy files

visual_data, pose_data = load_npy_data(data_path, n_sample, minibatch_sz, shuffle_data)

### Model Hyperparameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

load_model = False                          # If True, load a previously trained model from load_path. If False, train from scratch.

m1_inp_shape = visual_data.shape[1]         # modality 1 (default vision) input layer shape
m2_inp_shape = pose_data.shape[1]           # modality 2 (default pose) input layer shape

m1_layers = [1000, 300]                     # modality 1 layers shape
msi_layers = [100]                          # multi-modal integration layers shape; m2 currently has no hidden layers

m1_cause_init = [0.25, 0.25]                # the starting value for the inference process, whereby priors are updated according to evidence; modality 1
msi_cause_init = [0.25]                     # the starting value for the inference process, whereby priors are updated according to evidence; multi-modal integration

reg_m1_causes = [0.0, 0.0]                  # regularised error, disabled by default; modality 1
reg_m2_causes = [0.2, 0.2]                  # regularised error, disabled by default; modality 2
reg_msi_causes = [0.0]                      # regularised error, disabled by default; multi-modal integration

lr_m1_causes = [0.0004, 0.0004]             # learning rate for the inference process; modality 1
lr_msi_causes = [0.0004]                    # learning rate for the inference process; multi-modal integration

reg_m1_filters = [0.0, 0.0]                 # filters for regularised error, disabled by default; modality 1
reg_msi_filters = [0.0, 0.0]                # filters for regularised error, disabled by default; multi-modal integration

lr_m1_filters = [0.0001, 0.0001]            # learning rate for the inference process; modality 1
lr_msi_filters = [0.0001, 0.0001]           # learning rate for the inference process; multi-modal integration

class Network:
    def __init__(self, n_sample, minibatch_sz, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                 lr_m2_causes, lr_msi_causes, reg_m1_filters, reg_m2_filters, reg_msi_filters, lr_m1_filters,
                 lr_m2_filters, lr_msi_filters):

        self.m1_inp_shape = m1_inp_shape
        self.m2_inp_shape = m2_inp_shape
        self.m1_layers = m1_layers
        self.m2_layers = m2_layers
        self.msi_layers = msi_layers

        # create placeholders
        self.x_m1 = tf.placeholder(tf.float32, shape=[minibatch_sz, m1_inp_shape])
        self.x_m2 = tf.placeholder(tf.float32, shape=[minibatch_sz, m2_inp_shape])
        self.batch = tf.placeholder(tf.int32, shape=[])

        # create filters and cause for m1
        self.m1_filters = []
        self.m1_causes = []
        for i in range(len(self.m1_layers)):
            filter_name = 'm1_filter_%d' % i
            cause_name = 'm1_cause_%d' % i

            if i == 0:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_inp_shape])]
            else:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_layers[i-1]])]

            init = tf.constant_initializer(m1_cause_init[i])
            self.m1_causes += [tf.get_variable(cause_name, shape=[n_sample, self.m1_layers[i]], initializer=init)]

        # create filters and cause for msi
        self.msi_filters = []
        self.msi_causes = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                # add filters for m1
                filter_name = 'msi_m1_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m1_layers[-1]])]
                # add filters for m2
                filter_name = 'msi_m2_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m2_inp_shape])]
            else:
                filter_name = 'msi_filter_%d' % i
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.msi_layers[i - 1]])]

            cause_name = 'msi_cause_%d' % i
            init = tf.constant_initializer(msi_cause_init[i])
            self.msi_causes += [tf.get_variable(cause_name, shape=[n_sample, self.msi_layers[i]], initializer=init)]

        # compute predictions
        current_batch = tf.range(self.batch * minibatch_sz, (self.batch + 1) * minibatch_sz)
        # m1 predictions
        self.m1_minibatch = []
        self.m1_predictions = []
        for i in range(len(self.m1_layers)):
            self.m1_minibatch += [tf.gather(self.m1_causes[i], indices=current_batch, axis=0)]
            self.m1_predictions += [tf.nn.leaky_relu(tf.matmul(self.m1_minibatch[i], self.m1_filters[i]))]

        # msi predictions
        self.msi_minibatch = []
        self.msi_predictions = []
        for i in range(len(self.msi_layers)):
            self.msi_minibatch += [tf.gather(self.msi_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i]))]  # m1 prediction
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i+1]))]  # m2 prediction
            else:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i+1]))]

        # add ops for computing gradients for m1 causes and for updating weights
        self.m1_bu_error = []
        self.m1_update_filter = []
        self.m1_cause_grad = []
        for i in range(len(self.m1_layers)):
            if i == 0:
                self.m1_bu_error += [tf.losses.mean_squared_error(self.x_m1, self.m1_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.m1_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_minibatch[i - 1]), self.m1_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(self.m1_layers) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_predictions[i+1]), self.m1_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[0]), self.m1_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m1_causes[i] * (self.m1_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m1_causes[i])(self.m1_minibatch[i])
            self.m1_cause_grad += [tf.gradients([self.m1_bu_error[i], td_error, reg_error],
                                                          self.m1_minibatch[i])[0]]

            # ops for updating weights
            reg_error = reg_m1_filters[i] * (self.m1_filters[i] ** 2)
            m1_filter_grad = tf.gradients([self.m1_bu_error[i], reg_error], self.m1_filters[i])[0]
            self.m1_update_filter += [
                tf.assign_sub(self.m1_filters[i], lr_m1_filters[i] * m1_filter_grad)]

        # add ops for computing gradients for msi causes
        self.msi_bu_error = []
        self.msi_reg_error = []
        self.msi_update_filter = []
        self.msi_cause_grad = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_minibatch[-1]), self.msi_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_m2), self.msi_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.msi_reg_error += [reg_msi_causes[i] * (self.msi_minibatch[i] ** 2)]
                # self.msi_reg_error += [tf.keras.regularizers.l2(reg_msi_causes[i])(self.msi_minibatch[i])]
                if len(self.msi_layers) > 1:
                    raise NotImplementedError
                else:
                    self.msi_cause_grad += [
                        tf.gradients([self.msi_bu_error[i], self.msi_bu_error[i+1], self.msi_reg_error[i]],
                                               self.msi_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_msi_filters[i] * (self.msi_filters[i] ** 2)
                msi_filter_grad = tf.gradients([self.msi_bu_error[i], reg_error], self.msi_filters[i])[0]
                self.msi_update_filter += [
                    tf.assign_sub(self.msi_filters[i], lr_msi_filters[i] * msi_filter_grad)]
                reg_error = reg_msi_filters[i+1] * (self.msi_filters[i+1] ** 2)
                msi_filter_grad = tf.gradients([self.msi_bu_error[i+1], reg_error], self.msi_filters[i+1])[0]
                self.msi_update_filter += [
                    tf.assign_sub(self.msi_filters[i+1], lr_msi_filters[i+1] * msi_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.m1_update_cause = []
        self.m2_update_cause = []
        self.msi_update_cause = []
        with tf.control_dependencies(self.m1_cause_grad + self.msi_cause_grad):
            # m1 modality
            for i in range(len(self.m1_layers)):
                self.m1_update_cause += [tf.scatter_sub(self.m1_causes[i], indices=current_batch,
                                                                  updates=(lr_m1_causes[i] * self.m1_cause_grad[i]))]
            # msi modality
            for i in range(len(self.msi_layers)):
                self.msi_update_cause += [tf.scatter_sub(self.msi_causes[i], indices=current_batch,
                                                                   updates=(lr_msi_causes[i] * self.msi_cause_grad[i]))]


def train():
    tf.compat.v1.reset_default_graph()

    completed_epoch = 0

    net = Network(n_sample, minibatch_sz, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                  lr_m2_causes, lr_msi_causes, reg_m1_filters, reg_m2_filters, reg_msi_filters, lr_m1_filters,
                  lr_m2_filters, lr_msi_filters)

    saver = tf.train.Saver()
    cause_epoch = 50
    config = tf.ConfigProto(device_count={'GPU': 1})
    with tf.Session(config=config) as sess:
        if load_model is True:
            saver.restore(sess, '%s/main.ckpt' % load_path)
        else:
            sess.run(tf.global_variables_initializer())

        if load_model is True:
            m1_epoch_loss = np.load('%s/m1_epoch_loss.npy' % load_path)
            assert completed_epoch == m1_epoch_loss.shape[0], 'Value of completed_epoch is incorrect'

            m1_epoch_loss = np.vstack([m1_epoch_loss, np.zeros((n_epoch, len(m1_layers)))])
            
            msi_epoch_loss = np.vstack(
                [np.load('%s/msi_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(msi_layers) + 1))])

            m1_avg_activity = np.vstack(
                [np.load('%s/m1_avg_activity.npy' % load_path), np.zeros((n_epoch, len(m1_layers)))])
            msi_avg_activity = np.vstack(
                [np.load('%s/msi_avg_activity.npy' % load_path), np.zeros((n_epoch, len(msi_layers)))])
        else:
            m1_epoch_loss = np.zeros((n_epoch, len(m1_layers)))
            m2_epoch_loss = np.zeros((n_epoch, len(m2_layers)))
            msi_epoch_loss = np.zeros((n_epoch, len(msi_layers) + 1))

            m1_avg_activity = np.zeros((n_epoch, len(m1_layers)))
            m2_avg_activity = np.zeros((n_epoch, len(m2_layers)))
            msi_avg_activity = np.zeros((n_epoch, len(msi_layers)))

        for i in range(n_epoch):
            current_epoch = completed_epoch + i

            n_batch = visual_data.shape[0] // minibatch_sz
            for j in range(n_batch):
                visual_batch = visual_data[(j*minibatch_sz):((j+1)*minibatch_sz), :]
                pose_batch = pose_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]

                # update causes
                for k in range(cause_epoch):
                    m1_cause, msi_cause, m1_grad, msi_reg_error = sess.run(
                        [net.m1_update_cause, net.msi_update_cause,
                        net.m1_cause_grad, net.msi_reg_error],
                        feed_dict={net.x_m1: visual_batch, net.x_m2: pose_batch, net.batch: j})

                if save_m1_causes:

                    np.save(cause_save_path + '/m1/epoch{}_batch{}_cause_0'.format(i, j), m1_cause[0])
                    np.save(cause_save_path + '/m1/epoch{}_batch{}_cause_1'.format(i, j), m1_cause[1])

                if save_msi_causes:

                    np.save(cause_save_path + '/msi/epoch{}_batch{}'.format(i, j), msi_cause[1])

                # (optional) save reconstructions to diagnose training issues
                m1_reconstruction, msi_reconstruction = sess.run([net.m1_predictions[0], net.msi_predictions[1]],
                                                                feed_dict={net.x_m1: visual_batch, net.x_m2: pose_batch, net.batch: j})

                if save_m1_reconstructions:

                    np.save(reconstruction_save_path + '/m1/epoch{}_batch{}_reconstruction.npy'.format(i, j), m1_reconstruction)

                if save_msi_reconstructions:

                    np.save(reconstruction_save_path + '/msi/epoch{}_batch{}_reconstruction.npy'.format(i, j), msi_reconstruction)

                # update weights
                _, _, m1_error, msi_error, m1_filter, msi_filter = sess.run(
                    [net.m1_update_filter, net.msi_update_filter,
                     net.m1_bu_error, net.msi_bu_error,
                     net.m1_filters, net.msi_filters],
                    feed_dict={net.x_m1: visual_batch, net.x_m2: pose_batch, net.batch: j})

                # record maximum reconstruction error on the entire data
                m1_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > m1_epoch_loss[current_epoch, l]
                                                   else m1_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(m1_error)]
                msi_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                    if np.max(np.mean(item, axis=1)) > msi_epoch_loss[current_epoch, l]
                                                    else msi_epoch_loss[current_epoch, l]
                                                    for l, item in enumerate(msi_error)]

            # track average activity in inferred causes
            m1_avg_activity[current_epoch, :] = [np.mean(item) for item in m1_cause]
            msi_avg_activity[current_epoch, :] = [np.mean(item) for item in msi_cause]

            print('(%d) M1:%s (%s), MSI:%s (%s)' % (
                i, ', '.join(['%.8f' % elem for elem in m1_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in m1_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in msi_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in msi_avg_activity[current_epoch, :]])))

        # create the save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model and stats
        saver.save(sess, '%s/main.ckpt' % save_path)
        np.save('%s/m1_epoch_loss.npy' % save_path, m1_epoch_loss)
        np.save('%s/msi_epoch_loss.npy' % save_path, msi_epoch_loss)
        np.save('%s/m1_avg_activity.npy' % save_path, m1_avg_activity)
        np.save('%s/msi_avg_activity.npy' % save_path, msi_avg_activity)


if __name__ == '__main__':
    starttime = time.time()
    train()
    endtime = time.time()

    print ('Time taken: %f' % ((endtime - starttime) / 3600))

model_path = ' ' # Change this path to select saved model weights to load into the inference network

num_test_samps = 3000

error_criterion = np.array([1e-3, 1e-3, 3e-3, 3e-3]) # Corresponds to m1 layers [0, 1] and msi layers [0, 1]

max_iter = 500 # If causal inference doesn't get the prediction within the error criterion, timeout after X iterations

class InferenceNetwork:
    def __init__(self, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                 lr_m2_causes, lr_msi_causes, available_modality='both'):
        self.m1_inp_shape = m1_inp_shape
        self.m2_inp_shape = m2_inp_shape
        self.m1_layers = m1_layers
        self.m2_layers = m2_layers
        self.msi_layers = msi_layers

        # create placeholders
        self.x_m1 = tf.placeholder(tf.float32, shape=[1, m1_inp_shape])
        self.x_m2 = tf.placeholder(tf.float32, shape=[1, m2_inp_shape])

        # create filters and cause for m1
        self.m1_filters = []
        self.m1_causes = []
        for i in range(len(self.m1_layers)):
            filter_name = 'm1_filter_%d' % i
            cause_name = 'm1_cause_%d' % i

            if i == 0:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_inp_shape])]
            else:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_layers[i-1]])]

            init = tf.constant_initializer(m1_cause_init[i])
            self.m1_causes += [tf.get_variable(cause_name, shape=[1, self.m1_layers[i]], initializer=init)]

        # create filters and cause for msi
        self.msi_filters = []
        self.msi_causes = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                # add filters for m1
                filter_name = 'msi_m1_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m1_layers[-1]])]
                # add filters for m2
                filter_name = 'msi_m2_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m2_inp_shape])]
            else:
                filter_name = 'msi_filter_%d' % i
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.msi_layers[i - 1]])]

            cause_name = 'msi_cause_%d' % i
            init = tf.constant_initializer(msi_cause_init[i])
            self.msi_causes += [tf.get_variable(cause_name, shape=[1, self.msi_layers[i]], initializer=init)]

        # m1 predictions
        self.m1_predictions = []
        for i in range(len(self.m1_layers)):
            self.m1_predictions += [tf.nn.leaky_relu(tf.matmul(self.m1_causes[i], self.m1_filters[i]))]

        # msi predictions
        self.msi_predictions = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i]))]  # m1 prediction
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i+1]))]  # m2 prediction
            else:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i+1]))]

        # add ops for computing gradients for m1 causes and for updating weights
        self.m1_bu_error = []
        self.m1_update_filter = []
        self.m1_cause_grad = []
        for i in range(len(self.m1_layers)):
            if i == 0:
                self.m1_bu_error += [tf.losses.mean_squared_error(self.x_m1, self.m1_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.m1_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_causes[i - 1]), self.m1_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(self.m1_layers) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_predictions[i+1]), self.m1_causes[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[0]), self.m1_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m1_causes[i] * (self.m1_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m1_causes[i])(self.m1_minibatch[i])
            self.m1_cause_grad += [tf.gradients([self.m1_bu_error[i], td_error, reg_error],
                                                          self.m1_causes[i])[0]]

        # add ops for computing gradients for msi causes
        self.msi_bu_error = []
        self.msi_reg_error = []
        self.msi_update_filter = []
        self.msi_cause_grad = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_causes[-1]), self.msi_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.x_m2), self.msi_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.msi_reg_error += [reg_msi_causes[i] * (self.msi_causes[i] ** 2)]
                # self.msi_reg_error += [tf.keras.regularizers.l2(reg_msi_causes[i])(self.msi_minibatch[i])]
                if len(self.msi_layers) > 1:
                    raise NotImplementedError
                else:
                    if available_modality is 'both':
                        self.msi_cause_grad += [
                            tf.gradients([self.msi_bu_error[i], self.msi_bu_error[i+1], self.msi_reg_error[i]],
                                                   self.msi_causes[i])[0]]
                    elif available_modality is 'visual':
                        self.msi_cause_grad += [tf.gradients([self.msi_bu_error[i], self.msi_reg_error[i]],
                                                             self.msi_causes[i])[0]]
                    elif available_modality is 'pose':
                        self.msi_cause_grad += [tf.gradients([self.msi_bu_error[i + 1], self.msi_reg_error[i]],
                                                             self.msi_causes[i])[0]]

            else:
                raise NotImplementedError

        # add ops for updating causes
        self.m1_update_cause = []
        self.msi_update_cause = []
        with tf.control_dependencies(self.m1_cause_grad + self.msi_cause_grad):
            # m1 modality
            for i in range(len(self.m1_layers)):
                self.m1_update_cause += [tf.assign_sub(self.m1_causes[i], (lr_m1_causes[i] * self.m1_cause_grad[i]))]

            # msi modality
            for i in range(len(self.msi_layers)):
                self.msi_update_cause += [tf.assign_sub(self.msi_causes[i], (lr_msi_causes[i] * self.msi_cause_grad[i]))]

def init_network(model_path, available_modality='all'):
    tf.reset_default_graph()

    net = InferenceNetwork(m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                  lr_m2_causes, lr_msi_causes, available_modality)

    saver = tf.compat.v1.train.Saver(net.m1_filters + net.msi_filters)
    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver.restore(sess, '%smain.ckpt' % model_path)

    return sess, net

def infer_repr(sess, net, max_iter=1, error_criterion=0.0001, visual_data=None, pose_data=None, verbose=False,
               available_modality='all'):
    iter = 1

    while True:
        # infer representations
        m1_cause, msi_cause, m1_error, msi_error, m1_pred, msi_pred, m1_filter, msi_filter = sess.run(
                [net.m1_update_cause, net.msi_update_cause,
                 net.m1_bu_error, net.msi_bu_error,
                 net.m1_predictions, net.msi_predictions,
                 net.m1_filters, net.msi_filters],
                 feed_dict={net.x_m1: visual_data, net.x_m2: pose_data})
        if available_modality is 'both':
            m1_epoch_loss = [np.mean(item) for item in m1_error]
            #m2_epoch_loss = [np.mean(item) for item in m2_error]
            msi_epoch_loss = [np.mean(item) for item in msi_error]
        elif available_modality is 'visual':
            m1_epoch_loss = [np.mean(item) for item in m1_error]
            #m2_epoch_loss = [np.NINF, np.NINF]
            msi_epoch_loss = [np.mean(item) for item in msi_error]
        elif available_modality is 'pose':
            m1_epoch_loss = [np.NINF, np.NINF]
            #m2_epoch_loss = [np.mean(item) for item in m2_error]
            msi_epoch_loss = [np.mean(item) for item in msi_error]

        if (np.all(np.array(m1_epoch_loss + msi_epoch_loss) < error_criterion)) or (iter >= max_iter):
            if verbose:
                print_str = ', '.join(['%.8f' % elem for elem in m1_epoch_loss + msi_epoch_loss])
                print ('(%d) %s' % (iter, print_str))
            break
        else:
            iter += 1

    # reconstruct the missing modality
    recon_pose = np.dot(msi_cause[0], msi_filter[1])
    recon_vis = np.dot(msi_cause[0], msi_filter[0])
    for l in range(len(m1_filter), 0, -1):
        recon_vis = np.dot(recon_vis, m1_filter[l - 1])

    return msi_cause, recon_vis, recon_pose

def run_inference(sess, net, visual_data, pose_data, representations_save_path, avail_modality = 'both'):

    print("Modality: {}, Inferences will be saved to: {}".format(avail_modality, representations_save_path))

    print(visual_data.shape)
    print(pose_data.shape)

    assert visual_data.shape[0] == pose_data.shape[0]

    recon_temp_vis = np.zeros([ visual_data.shape[0], 45, 80, 3 ])
    recon_temp_tac = np.zeros([ pose_data.shape[0] , 180 ])
    representations = np.zeros([pose_data.shape[0], 100])

    sess.run(tf.compat.v1.variables_initializer(net.m1_causes + net.msi_causes))

    for j in range (visual_data.shape[0]):
        visual_input = visual_data[None, j]
        pose_input = pose_data[None, j]
        if avail_modality is 'visual':
            pose_input = np.zeros([1, 180])
        elif avail_modality is 'pose':
            visual_input = np.zeros([1, 10800])
        reps, recon_vis, recon_tac = infer_repr(sess, net, max_iter, error_criterion, visual_input, pose_input, True, avail_modality)
        representations[j, :] = reps[0]
        recon_temp_vis[j, :] = recon_vis.reshape(45,80,3) # reform into image
        recon_temp_tac[j, :] = recon_tac

    print('Done!')

    np.save(representations_save_path + 'representations_extra.npy', representations)
    sio.savemat(representations_save_path + 'reps_extra.mat', {'reps':representations})

    np.save(representations_save_path + 'reconstructions_visual_extra.npy', recon_temp_vis)
    sio.savemat(representations_save_path + 'recon_vis_extra.mat', {'recon_vis':recon_temp_vis})

    np.save(representations_save_path + 'reconstructions_head_direction_extra.npy', recon_temp_tac)
    sio.savemat(representations_save_path + 'recon_tac_extra.mat', {'recon_tac':recon_temp_tac})

def generate_representations(shuffle):

    root_test_path = ' ' # Point this to the folder containing all the test data folders
    root_representations_path = ' ' # Point this to where you want the predictions/representations to be output

    if True:
        dataset = ' ' # Change this to point to the data folder you want to draw data (images.npy and networkOutput_gaussianied.npy)
    # Alternatively, swap this to: "for dataset in (<comma-seperated folder names>):" for multiple datasets

        test_set = root_test_path + "{}".format(dataset)
        representations_save_path = root_representations_path + "{}/visual/".format(dataset)

        visual_data, pose_data = load_npy_data(test_set, num_test_samps, minibatch_sz, shuffle = shuffle)#load_mat_data(test_set, shuffle = shuffle)

        sess, net = init_network(model_path, available_modality = 'visual')

        print("Dataset {}".format(dataset))

        run_inference(sess, net, visual_data, pose_data, representations_save_path, avail_modality = 'visual')

starttime = time.time()

generate_representations(shuffle = False)

endtime = time.time()

print ('Time taken: %f' % ((endtime - starttime) / 3600))
