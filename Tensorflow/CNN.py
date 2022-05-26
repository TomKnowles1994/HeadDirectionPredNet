import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def load_npy_data(data_path, sample_count, images_file = '/images.npy', pose_file = '/networkOutput_gaussianised.npy', offset = 0, shuffle=True):

    img = np.load(data_path + images_file)
    
    print(img.shape)
    
    if img.shape[0] > sample_count:

        img = np.load(data_path + images_file)[offset:sample_count+offset]

        print(img.shape)

    pose_data = np.load(data_path + pose_file)

    print(pose_data.shape)

    if pose_data.shape[0] > sample_count:

        pose_data = np.load(data_path + pose_file)[offset:sample_count+offset]

        print(pose_data.shape)

    if img.shape[0] != pose_data.shape[0]:

        min_sample_count = min(img.shape[0], pose_data.shape[0])

        img = img[offset:min_sample_count+offset]
        print(img.shape)

        pose_data = pose_data[offset:min_sample_count+offset]
        print(pose_data.shape)

    if shuffle:
        # shuffle sequence of data but maintain visual-pose alignment
        img, pose_data = shuffle_in_sync(img, pose_data)

    return img, pose_data

def shuffle_in_sync(visual_data, pose_data):

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_pose = visual_data[shared_indices], pose_data[shared_indices]

    return shuffled_visual, shuffled_pose

data_path = ' ' # Point this to the training data folder

save_path = ' ' # Point this to where the checkpoints are to be saved

images, head_direction = load_npy_data(data_path, 3000, pose_file = '/networkOutput_gaussianised.npy', shuffle = False)

def build_model():

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(180))
    model.compile(loss='mse')

    model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="mse"
    )

    return model

def one_hot_encode(data_path, n_sample):

    poses = np.load(data_path + '/networkOutput_gaussianised.npy')

    one_hot_poses = np.zeros_like(poses)

    one_hot_poses[:, np.argmax(poses, axis = 1)] = 1

    return poses

model = build_model()

val_data_path = ' ' # Point this to the validation set

val_images, val_head_direction = load_npy_data(val_data_path, 2000, pose_file = '/networkOutput_gaussianised.npy', offset = 3000, shuffle = False)

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

model.fit(x = images, y = head_direction, validation_data = (val_images, val_head_direction), batch_size = 10, epochs = 50, callbacks = [lr_schedule])

model.save_weights(save_path + '/main.ckpt')

model.load_weights(save_path + '/main.ckpt')

if True:
    dataset = ' ' # Point to the dataset folder

# Alternatively: "for dataset in (<comma-seperated dataset folders>)" for multiple folders:

    print("Creating Predictions for {} dataset".format(dataset))

    data_path = ' ' # Point to dataset folder

    images, _ = load_npy_data(data_path, 3000, pose_file = '/networkOutput_gaussianised.npy', shuffle = False)

    predictions = model.predict(x = images)

    predictions_save_path = ' ' # Output predictions here

    np.save(predictions_save_path + "/visual/predictions_conv.npy", predictions)
