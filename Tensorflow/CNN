import time, os, imghdr, random
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
#from keras_tuner import RandomSearch, applications

def load_npy_data(data_path, sample_count, images_file = '/images.npy', pose_file = '/networkOutput_gaussianised.npy', offset = 0, shuffle=True):

    img = np.load(data_path + images_file)
    
    print(img.shape)
    
    if img.shape[0] > sample_count:

        img = np.load(data_path + images_file)[offset:sample_count+offset]

        print(img.shape)

    #img = img.reshape(img.shape[0], 10800) # flatten

    pose_data = np.load(data_path + pose_file)#[1:]#np.load(data_path + '/networkOutput.npy').T
    #print(pose_data.shape)

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
    #assert visual_data.shape[0] == pose_data.shape[0]

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_pose = visual_data[shared_indices], pose_data[shared_indices]

    return shuffled_visual, shuffled_pose

data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_trainingset'

save_path = 'C:/Users/Thomas/Downloads/HBP/model_checkpoints/landmarks_vh/whiskeye_head_direction_full_convnet_refined'

images, head_direction = load_npy_data(data_path, 3000, pose_file = '/networkOutput_gaussianised.npy', shuffle = False)

def build_model():

    model = keras.Sequential()
    #model.add(keras.layers.Dropout(.0, input_shape=(45, 80, 3)))
    #model.add(keras.layers.Conv2D(128, (8, 8), activation='relu'))#, input_shape=(45, 80, 3)))
    #model.add(keras.layers.MaxPooling2D((4, 4)))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dropout(.1))
    #model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(180))
    model.compile(loss='mse')

    model.compile(
    optimizer=keras.optimizers.Adam(1e-6),
    loss="mse"
    )

    return model

def one_hot_encode(data_path, n_sample):

    poses = np.load(data_path + '/networkOutput_gaussianised.npy')#[0:n_sample]

    one_hot_poses = np.zeros_like(poses)

    one_hot_poses[:, np.argmax(poses, axis = 1)] = 1

    return poses

#head_direction_one_hot = one_hot_encode(data_path, 1000)

#hyperresnet = applications.HyperResNet(include_top=False, input_shape=(45,80,3), input_tensor=None, classes=None)

#tuner = RandomSearch(hyperresnet, objective='loss', max_trials=25)

#tuner = RandomSearch(build_model, objective='loss', max_trials=25)

#tuner.search_space_summary()

#tuner.search(images, head_direction, batch_size = 10, epochs=1)

#models = tuner.get_best_models(num_models=5)

model = build_model()

data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_trainingset'

val_data_path = 'C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_rotating_distal'

val_images, val_head_direction = load_npy_data(val_data_path, 2000, pose_file = '/networkOutput_gaussianised.npy', offset = 3000, shuffle = False)

#val_head_direction_one_hot = one_hot_encode(data_path, 500)

def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
        #return 1e-6
lr_schedule = keras.callbacks.LearningRateScheduler(scheduler)

model.fit(x = images, y = head_direction, validation_data = (val_images, val_head_direction), batch_size = 10, epochs = 50, callbacks = [lr_schedule])

model.save_weights(save_path + '/main.ckpt')

model.load_weights(save_path + '/main.ckpt')

#for dataset in (1,6):#range(1,21):#(1,2,3,4,5,6,9,10,13,15,16,19):
#dataset = "rotating_distal"
#if True:
#for dataset in ("rotating_distal", "rotating_proximal", "circling_distal", "circling_proximal", "random_distal", "random_proximal", "cogarch"):
#for dataset in ("random_distal_2", "random_distal_3", "random_distal_4", "random_distal_5"):
for dataset in ("rotating_distal", "circling_distal", "random_distal","random_distal_2", "random_distal_3", "random_distal_4", "random_distal_5"):

    #model.load_weights(save_path + '/main.ckpt')

    print("Creating Predictions for {} dataset".format(dataset))

    data_path = "C:/Users/Thomas/Downloads/HBP/multimodalplacerecognition_datasets/whiskeye_head_direction_{}".format(dataset)

    images, _ = load_npy_data(data_path, 3000, pose_file = '/networkOutput_gaussianised.npy', shuffle = False)

    predictions = model.predict(x = images)

    predictions_save_path = "C:/Users/Thomas/Downloads/HBP/representations/NRP/whiskeye_head_direction_{}_convnet_refined".format(dataset)

    np.save(predictions_save_path + "/visual/predictions_conv.npy", predictions)
