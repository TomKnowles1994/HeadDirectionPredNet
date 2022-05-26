import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skimage.util import img_as_float, img_as_ubyte
import scipy.io as sio

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import InputLayer, Dense, Concatenate
from tensorflow.keras.losses import MeanSquaredError, Reduction
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import Mean

from PIL import Image
from collections import OrderedDict

from numpy.random import permutation

from tensorflow.python.keras import backend as K

tf.compat.v1.reset_default_graph()

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1} )
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

K.set_session(sess)

def shuffle_in_sync(visual_data, pose_data):

    assert visual_data.shape[0] == pose_data.shape[0]

    print(visual_data.shape)
    print(pose_data.shape)

    shared_indices = permutation(visual_data.shape[0])

    shuffle_visual, shuffle_tactical = visual_data[shared_indices], pose_data[shared_indices]

    return shuffle_visual, shuffle_tactical

def load_npy_data(data_path, shuffle=True):

    img = np.load(data_path + '/images.npy')
    img = img.reshape(img.shape[0], 10800)
    pose_data = np.load(data_path + '/networkOutput_gaussianised.npy')

    if shuffle:
        # shuffle sequence of data but maintain visual-pose alignment
        img, pose_data = shuffle_in_sync(img, pose_data)

    return img, pose_data

def preprocess_pose_data(pose_data):
    global scaler
    scaler = MinMaxScaler(copy=False)
    scaler.fit(pose_data)
    scaler.transform(pose_data)

    return pose_data

class JMVAE_kl(Model):
    def __init__(self, input_dim_m1, encoder_hidden_dim_m1, decoder_hidden_dim_m1,
                 input_dim_m2, encoder_hidden_dim_m2, decoder_hidden_dim_m2, latent_dim, beta, alpha):
        super(JMVAE_kl, self).__init__()
        self.input_dim_m1 = input_dim_m1
        self.encoder_hidden_dim_m1 = encoder_hidden_dim_m1
        self.decoder_hidden_dim_m1 = decoder_hidden_dim_m1
        self.input_dim_m2 = input_dim_m2
        self.encoder_hidden_dim_m2 = encoder_hidden_dim_m2
        self.decoder_hidden_dim_m2 = decoder_hidden_dim_m2
        self.latent_dim = latent_dim
        self.beta = beta
        self.alpha = alpha

        self.create_encoders()
        self.create_decoders()

    def create_encoders(self):
        # create unimodal M1 encoder
        unimodal_m1_inp = Input(shape=(self.input_dim_m1,))
        prev_layer_m1 = unimodal_m1_inp
        unimodal_m1_out = Dense(units=(self.latent_dim * 2))(prev_layer_m1)

        # create unimodal M2 encoder
        unimodal_m2_inp = Input(shape=(self.input_dim_m2,))
        prev_layer_m2 = unimodal_m2_inp
        for h in self.encoder_hidden_dim_m2:
            x = Dense(units=h, activation='relu')(prev_layer_m2)
            prev_layer_m2 = x
        unimodal_m2_out = Dense(units=(self.latent_dim * 2))(prev_layer_m2)

        # create multimodal M1 encoder
        multimodal_m1_inp = Input(shape=(self.input_dim_m1,))
        prev_layer_m1 = multimodal_m1_inp

        # create multimodal M2 encoder
        multimodal_m2_inp = Input(shape=(self.input_dim_m2,))
        prev_layer_m2 = multimodal_m2_inp
        for h in self.encoder_hidden_dim_m2:
            x = Dense(units=h, activation='relu')(prev_layer_m2)
            prev_layer_m2 = x

        concat_out = Concatenate()([prev_layer_m1, prev_layer_m2])
        multimodal_out = Dense(units=(self.latent_dim*2))(concat_out)
        self.encoder = Model(inputs=[unimodal_m1_inp, unimodal_m2_inp, multimodal_m1_inp, multimodal_m2_inp],
                             outputs=[unimodal_m1_out, unimodal_m2_out, multimodal_out])

    def create_decoders(self):
        # create unimodal M1 decoder
        unimodal_m1_inp = Input(shape=(self.latent_dim,))  # there is one input layer in case of decoder
        prev_layer_m1 = unimodal_m1_inp
        unimodal_m1_out = Dense(units=self.input_dim_m1, activation='sigmoid')(prev_layer_m1)

        # create unimodal M2 decoder
        unimodal_m2_inp = Input(shape=(self.latent_dim,))  # there is one input layer in case of decoder
        prev_layer_m2 = unimodal_m2_inp
        for h in self.decoder_hidden_dim_m2:
            x = Dense(units=h, activation='relu')(prev_layer_m2)
            prev_layer_m2 = x
        unimodal_m2_out = Dense(units=self.input_dim_m2, activation='sigmoid')(prev_layer_m2)

        # create multimodal decoders
        multimodal_inp = Input(shape=(self.latent_dim,)) # there is one input layer in case of decoder
        # create multimodal M1 decoder
        prev_layer_m1 = multimodal_inp
        multimodal_m1_out = Dense(units=self.input_dim_m1, activation='sigmoid')(prev_layer_m1)
        # create multimodal M2 decoder
        prev_layer_m2 = multimodal_inp
        for h in self.decoder_hidden_dim_m2:
            x = Dense(units=h, activation='relu')(prev_layer_m2)
            prev_layer_m2 = x
        multimodal_m2_out = Dense(units=self.input_dim_m2, activation='sigmoid')(prev_layer_m2)

        self.decoder = Model(inputs=[unimodal_m1_inp, unimodal_m2_inp, multimodal_inp],
                             outputs=[unimodal_m1_out, unimodal_m2_out, multimodal_m1_out, multimodal_m2_out])

    def encode(self, x):
        unimodal_m1_output, unimodal_m2_output, multimodal_output = self.encoder(x)
        # mean, logvar = tf.split(multimodal_output, num_or_size_splits=2, axis=1)
        return unimodal_m1_output, unimodal_m2_output, multimodal_output

    def decode(self, x):
        return self.decoder(x)

    def reparameterise(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return (eps * tf.exp(logvar * 0.5)) + mean


def compute_loss(model, x):
    uni_m1_output, uni_m2_output, multi_output = model.encode(x)
    # reparametrize unimodal m1 latent variable
    uni_m1_mean, uni_m1_logvar = tf.split(uni_m1_output, num_or_size_splits=2, axis=1)
    uni_m1_z = model.reparameterise(uni_m1_mean, uni_m1_logvar)
    # reparametrize unimodal m2 latent variable
    uni_m2_mean, uni_m2_logvar = tf.split(uni_m2_output, num_or_size_splits=2, axis=1)
    uni_m2_z = model.reparameterise(uni_m2_mean, uni_m2_logvar)
    # reparametrize multimodal latent variable
    mean, logvar = tf.split(multi_output, num_or_size_splits=2, axis=1)
    multi_z = model.reparameterise(mean, logvar)
    global x_prob
    x_prob = model.decode([uni_m1_z, uni_m2_z, multi_z])

    # reconstruction error for unimodal M1
    mse = MeanSquaredError(reduction=Reduction.NONE)
    recon_err_uni_m1 = mse(tf.expand_dims(x[0], axis=-1), tf.expand_dims(x_prob[0], axis=-1))  # add a dimension to x and x_prob as MSE returns one value per sample
    recon_err_uni_m1 = tf.reduce_mean(tf.reduce_sum(recon_err_uni_m1, axis=1), axis=0)

    # reconstruction error for unimodal M2
    mse = MeanSquaredError(reduction=Reduction.NONE)
    recon_err_uni_m2 = mse(tf.expand_dims(x[1], axis=-1), tf.expand_dims(x_prob[1], axis=-1))  # add a dimension to x and x_prob as MSE returns one value per sample
    recon_err_uni_m2 = tf.reduce_mean(tf.reduce_sum(recon_err_uni_m2, axis=1), axis=0)

    # reconstruction error for multimodal M1
    mse = MeanSquaredError(reduction=Reduction.NONE)
    recon_err_multi_m1 = mse(tf.expand_dims(x[2], axis=-1), tf.expand_dims(x_prob[2], axis=-1)) # add a dimension to x and x_prob as MSE returns one value per sample
    recon_err_multi_m1 = tf.reduce_mean(tf.reduce_sum(recon_err_multi_m1, axis=1), axis=0)

    # reconstruction error for multimodal M2
    mse = MeanSquaredError(reduction=Reduction.NONE)
    recon_err_multi_m2 = mse(tf.expand_dims(x[3], axis=-1), tf.expand_dims(x_prob[3], axis=-1)) # add a dimension to x and x_prob as MSE returns one value per sample
    recon_err_multi_m2 = tf.reduce_mean(tf.reduce_sum(recon_err_multi_m2, axis=1), axis=0)

    # KL-divergence between q(z|x, w) and N(0, 1)
    kl_div = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
    # NOTE: Multiplication with beta seems like a JMVAE specific thing
    kl_div = tf.reduce_mean(model.beta * tf.reduce_sum(kl_div, axis=1))

    # KL-divergence between q(z|x) and N(0, 1)
    m1_kl_div = -0.5 * (1 + uni_m1_logvar - tf.square(uni_m1_mean) - tf.exp(uni_m1_logvar))
    # NOTE: Multiplication with beta seems like a JMVAE specific thing
    m1_kl_div = tf.reduce_mean(model.beta * tf.reduce_sum(m1_kl_div, axis=1))

    # KL-divergence between q(z|w) and N(0, 1)
    m2_kl_div = -0.5 * (1 + uni_m2_logvar - tf.square(uni_m2_mean) - tf.exp(uni_m2_logvar))
    # NOTE: Multiplication with beta seems like a JMVAE specific thing
    m2_kl_div = tf.reduce_mean(model.beta * tf.reduce_sum(m2_kl_div, axis=1))

    # KL-divergence between q(z|x, w) and q(z|x)
    # The formula for KL-divergence between two gaussians (D_KL (p1 || p2)) is
    # 1 + (logvar1 - logvar2) - (var2_inverse * var1) - ((mean2 - mean1) * var2_inverse * (mean2 - mean1))
    # we will denote the three main components by a, b and c. We will compute each one-by-one
    a = logvar - uni_m1_logvar
    b = (1 / tf.exp(uni_m1_logvar)) * tf.exp(logvar)
    c = (uni_m1_mean - mean) * (1 / tf.exp(uni_m1_logvar)) * (uni_m1_mean - mean)
    m1_mm_kl_div = -0.5 * (1 + a - b - c)
    m1_mm_kl_div = tf.reduce_mean(model.alpha * tf.reduce_sum(m1_mm_kl_div, axis=1))

    # KL-divergence between q(z|x, w) and q(z|w)
    a = logvar - uni_m2_logvar
    b = (1 / tf.exp(uni_m2_logvar)) * tf.exp(logvar)
    c = (uni_m2_mean - mean) * (1 / tf.exp(uni_m2_logvar)) * (uni_m2_mean - mean)
    m2_mm_kl_div = -0.5 * (1 + a - b - c)
    m2_mm_kl_div = tf.reduce_mean(model.alpha * tf.reduce_sum(m2_mm_kl_div, axis=1))

    return recon_err_uni_m1 + recon_err_uni_m2 + recon_err_multi_m1 + recon_err_multi_m2 + kl_div + m1_kl_div + \
          m2_kl_div + m1_mm_kl_div + m2_mm_kl_div, recon_err_uni_m1, recon_err_uni_m2, m1_mm_kl_div, m2_mm_kl_div, \
          mean, uni_m1_mean, uni_m2_mean


def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, _, _, _, _, _, _, _ = compute_loss(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Data paths

tr_data_path = ' ' # Point to training data folder

ts_data_path =  ' ' # Point to test data folder

checkpoint_path = ' ' # Point to checkpoint folder

def train():

    # Data shape

    data_sz = 1000
    batch_sz = 10
    test_batch_sz = 10

    # load the data
    tr_visual_data, tr_pose_data = load_npy_data(tr_data_path, shuffle = False)
    ts_visual_data, ts_pose_data = load_npy_data(ts_data_path, shuffle = False)

    tr_visual_dataset = tf.data.Dataset.from_tensor_slices(tr_visual_data[:data_sz, :]).batch(batch_sz)
    tr_pose_dataset = tf.data.Dataset.from_tensor_slices(tr_pose_data[:data_sz, :]).batch(batch_sz)

    ts_visual_dataset = tf.data.Dataset.from_tensor_slices(ts_visual_data).batch(test_batch_sz)
    ts_pose_dataset = tf.data.Dataset.from_tensor_slices(ts_pose_data).batch(test_batch_sz)

    # create VAE for head direction and camera images
    input_dim_m1 = 63
    encoder_hidden_dim_m1 = [100]
    decoder_hidden_dim_m1 = [100]

    input_dim_m2 = 10800
    encoder_hidden_dim_m2 = [1000, 300]
    decoder_hidden_dim_m2 = [300, 1000]

    latent_dim = 100
    beta = 1  # strength of distribution regularization
    alpha = 0.05  # strength of KL-divergence between the distribution of latent variables
    
    vae =   JMVAE_kl(input_dim_m1, encoder_hidden_dim_m1, decoder_hidden_dim_m1,
            input_dim_m2, encoder_hidden_dim_m2, decoder_hidden_dim_m2, latent_dim, beta, alpha)

    optimizer = Adam(learning_rate=0.00005)

    # train the Multimodal VAE
    n_epochs = 5000
    for e in range(n_epochs):
        for train_x in zip(tr_pose_dataset, tr_visual_dataset, tr_pose_dataset, tr_visual_dataset):
            train_x = list(train_x)  # this step is not really required, but to be on safe side !!!
            train_step(vae, train_x, optimizer)

        # compute ELBO on test dataset
        for test_x in zip(ts_pose_dataset, ts_visual_dataset, ts_pose_dataset, ts_visual_dataset):
            test_x = list(test_x)
            vae_loss, m1_recon_loss, m2_recon_loss, m1_kl_div, m2_kl_div, reps, reps_m1, reps_m2 = compute_loss(vae, test_x)
        ELBO = -vae_loss.numpy()
        recon_m1_loss_val = m1_recon_loss.numpy()
        recon_m2_loss_val = m2_recon_loss.numpy()

        print('Epoch: {}; ELBO: {}; Reconstruction (M1, M2): {}, {}, KL_div: {}, {}'.format(e, ELBO, recon_m1_loss_val,
                                                                                            recon_m2_loss_val,
                                                                                            m1_kl_div, m2_kl_div))
    
    # Save the trained model
    vae.save_weights(checkpoint_path + 'main.ckpt')

def test_and_generate(  test_set, reconstructions_save_path, representations_save_path, available_modality = "both", 
                        output_pngs = True, output_npys = True, output_matlab = True):

    # create VAE for whiskers and camera images
    input_dim_m1 = 63
    encoder_hidden_dim_m1 = [100]#[50, 20]#[100, 75]#[100]  # [50, 25]
    decoder_hidden_dim_m1 = [100]#[20, 50]#[75, 100]#[100]  # [25, 50]

    input_dim_m2 = 10800
    encoder_hidden_dim_m2 = [1000, 300]  # [5000, 1000, 100]
    decoder_hidden_dim_m2 = [300, 1000]  # [100, 1000, 5000]

    latent_dim = 100
    beta = 1  # strength of distribution regularization
    alpha = 0.05  # strength of KL-divergence between the distribution of latent variables
    model = JMVAE_kl(input_dim_m1, encoder_hidden_dim_m1, decoder_hidden_dim_m1,
                               input_dim_m2, encoder_hidden_dim_m2, decoder_hidden_dim_m2, latent_dim, beta, alpha)

    optimizer = Adam(learning_rate=0.00005)

    model.load_weights(' ') # Point to saved weight file (checkpoint)

    test_batch_sz = 500

    ts_visual_data, ts_pose_data = load_npy_data(test_set, shuffle = False)

    ts_visual_data = ts_visual_data[:test_batch_sz]
    ts_pose_data = ts_pose_data[:test_batch_sz]

    if available_modality == 'pose':
        ts_visual_data = np.zeros_like(ts_visual_data)
    
    elif available_modality == 'visual':
        ts_pose_data = np.zeros_like(ts_pose_data)

    ts_visual_dataset = tf.data.Dataset.from_tensor_slices(ts_visual_data).batch(1)
    ts_pose_dataset = tf.data.Dataset.from_tensor_slices(ts_pose_data).batch(1)

    # Output representations, feeding test samples one-by-one

    representations_unimodal_vision = np.zeros(shape=(test_batch_sz,latent_dim))
    representations_unimodal_pose = np.zeros(shape=(test_batch_sz,latent_dim))
    representations_multimodal = np.zeros(shape=(test_batch_sz,latent_dim))

    reconstructions_unimodal_pose = np.zeros(shape=(test_batch_sz,input_dim_m1))
    reconstructions_unimodal_vision = np.zeros(shape=(test_batch_sz,45,80,3))
    reconstructions_multimodal_pose = np.zeros(shape=(test_batch_sz,input_dim_m1))
    reconstructions_multimodal_vision = np.zeros(shape=(test_batch_sz,45,80,3))

    sample_index = 0

    for sample in zip(ts_pose_dataset, ts_visual_dataset, ts_pose_dataset, ts_visual_dataset):

        sample = list(sample)
        _, _, _, _, _, reps, reps_m1, reps_m2 = compute_loss(model, sample)

        representations_unimodal_vision[sample_index,:]         = reps_m2
        representations_unimodal_pose[sample_index,:]           = reps_m1
        representations_multimodal[sample_index,:]              = reps

        reconstructions_unimodal_pose[sample_index,:]           = x_prob[0].numpy()
        reconstructions_unimodal_vision[sample_index,:,:,:]     = x_prob[1].numpy().reshape(45,80,3)
        reconstructions_multimodal_pose[sample_index,:]         = x_prob[2].numpy()
        reconstructions_multimodal_vision[sample_index,:,:,:]   = x_prob[3].numpy().reshape(45,80,3)

        sample_index+=1

    print("Representation generation done")

    if output_pngs == True:

        for image in range(test_batch_sz):

            Image.fromarray((reconstructions_unimodal_vision[image,:,:,:] * 255).astype(np.uint8)).convert('RGB').save(reconstructions_save_path + 'unimodal{}.png'.format(image))
            Image.fromarray((reconstructions_multimodal_vision[image,:,:,:] * 255).astype(np.uint8)).convert('RGB').save(reconstructions_save_path + 'multimodal{}.png'.format(image))

    if output_npys == True:

        np.save('{}/reconstructions_unimodal_pose.npy'.format(representations_save_path), reconstructions_unimodal_pose)
        np.save('{}/reconstructions_unimodal_visual.npy'.format(representations_save_path), reconstructions_unimodal_vision)
        np.save('{}/reconstructions_multimodal_pose.npy'.format(representations_save_path), reconstructions_multimodal_pose)
        np.save('{}/reconstructions_multimodal_visual.npy'.format(representations_save_path), reconstructions_multimodal_vision)

    if output_matlab == True:

        sio.savemat(representations_save_path + 'reconstructions_unimodal_pose.mat', {'reconstructions_unimodal_pose':reconstructions_unimodal_pose})
        sio.savemat(representations_save_path + 'reconstructions_unimodal_visual.mat', {'reconstructions_unimodal_visual':reconstructions_unimodal_vision})
        sio.savemat(representations_save_path + 'reconstructions_multimodal_pose.mat', {'reconstructions_multimodal_pose':reconstructions_multimodal_pose})
        sio.savemat(representations_save_path + 'reconstructions_multimodal_visual.mat', {'reconstructions_multimodal_visual':reconstructions_multimodal_vision})

train()

def generate_mupnet_comparison_data():

    K.clear_session()
    tf.compat.v1.reset_default_graph()
    
    # These device settings may not be appropriate for others' setups
    
    config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1} )
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    K.set_session(sess)
    
    if True:
        dataset = ' ' # Point to data folder

    # Alternatively: "for dataset in (<comma-seperated dataset folders>):" for multiple datasets

        test_set = '{}'.format(dataset) # Fully qualified filepath, minus specific data folder
        reconstructions_save_path = '{}'.format(dataset) # Fully qualified filepath, assuming reconstructions output folder named the same as dataset
        representations_save_path = '{}'.format(dataset) # Fully qualified filepath, assuming same name as dataset. This will usually be the same as reconstructions save path

        test_and_generate(  test_set, reconstructions_save_path, representations_save_path, available_modality = "visual", 
                            output_pngs = False, output_npys = True, output_matlab = True)

    K.clear_session()


generate_mupnet_comparison_data()
