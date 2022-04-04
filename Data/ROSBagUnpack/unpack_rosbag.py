import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import os
from glob import glob
from natsort import natsorted

halve_and_merge = False              # True if you want the Whiskeye arrays cut from the image and the remainders merged. This preserves image dimensions, but gives more information about the world.
preserve_pose_index = True          # True if you want to keep the original pose index as a column in the final dataset. False if you want MultiPredNet-ready data.
preserve_pose_timestamps = True     # True if you want to keep the pose timestamps in the final dataset. False if you want MultiPredNEt-ready data.

# Convert images to .npy

print("Converting images to .npy")

if halve_and_merge is False:

        # Get timestamps from image csv

        timestamps = pd.read_csv("raw_images_cam0.csv", usecols = ['%time'])

        jpg_iterator = natsorted(glob("cam0/*.jpg"))

        raw_images = []
        images = [] # Normalised and resized to [45, 80]

        for jpg in jpg_iterator:

                image = cv.imread(jpg).astype(np.float32)[:,:,::-1]
                raw_images.append(image)

                image = cv.resize(image, (80, 45))
                image = image / 255
                images.append(image)

        cv.imwrite("sample.png", image * 255)

        #raw_images_npy = np.array(raw_images)
        #images_npy = np.array(images)

        #np.save('raw_images.npy', np.array(raw_images))
        np.save('images.npy', np.array(images))

if halve_and_merge:

        # Get timestamps from image csv

        timestamps = pd.read_csv("raw_images_cam0.csv", usecols = ['%time'])

        left_jpg_iterator = iglob("cam0/*.jpg")
        right_jpg_iterator = iglob("cam1/*.jpg")

        #raw_images_left = []
        #raw_images_right = []
        images = [] # Normalised and resized to [45, 80]

        for left, right in zip(left_jpg_iterator, right_jpg_iterator):

                image_left = cv.imread(left).astype(np.float32)[:,:,::-1]
                image_right = cv.imread(right).astype(np.float32)[:,:,::-1]
                #raw_images_left.append(image_left)
                #raw_images_right.append(image_right)

                image_left = cv.resize(image_left, (80, 45))
                image_right = cv.resize(image_right, (80, 45))
                image_left = image_left[:,:40]
                image_right = image_right[:,40:]
                image_left = image_left / 255
                image_right = image_right / 255

                image = np.concatenate((image_left, image_right), axis=1)
                images.append(image)

        cv.imwrite("merged_sample.png", image * 255)

        #raw_images_left_npy = np.array(raw_images_left)
        #raw_images_right_npy = np.array(raw_images_right)

        images_npy = np.array(images)

        #np.save('raw_images_0.npy', raw_images_left_npy)
        #np.save('raw_images_1.npy', raw_images_right_npy)
        np.save('images.npy', images_npy)


# Save subset of head poses that match image timestamps

print("Converting peak protraction poses to .npy")

pose = pd.read_csv("raw_pose.csv")

pose_indices = np.searchsorted(pose['%time'], timestamps['%time']).flatten()

pose_in_sync = pose.iloc[pose_indices]

pose_in_sync = pose_in_sync.reset_index()

if preserve_pose_timestamps is True and preserve_pose_index is not True:

        pose_in_sync = pose_in_sync.iloc[:,1:]

elif preserve_pose_index is True and preserve_pose_timestamps is not True:

        pose_in_sync = pose_in_sync.iloc[:,0:]

        pose_in_sync.drop(columns = '%time', inplace = True)

elif preserve_pose_index is True and preserve_pose_timestamps is True:

        pose_in_sync = pose_in_sync.iloc[:,0:]

elif preserve_pose_index is not True and preserve_pose_timestamps is not True:

        pose_in_sync = pose_in_sync.iloc[:,2:]

#print(pose_in_sync.columns)

pose_in_sync.rename(columns = {"field.x": "positionX", "field.y": "positionY", "field.theta": "theta"}, inplace = True)

pose_in_sync.to_csv('pose.csv', index = False)

np.save('poses.npy', pose_in_sync)

#print(pose_in_sync.shape)


# Save subset of whisker theta angle that match image timestamps

print("Converting peak protraction theta to .npy")

theta = pd.read_csv("raw_theta.csv")

theta_indices = np.searchsorted(theta['%time'], timestamps['%time']).flatten()

theta_in_sync = theta.iloc[theta_indices].reset_index()

theta_subsampled = theta_in_sync.iloc[:, 3::10]

theta_subsampled.rename(columns = {column: "whisker{}".format(theta_subsampled.columns.get_loc(column)//2+1) for column in theta_subsampled.columns}, inplace = True)

theta_subsampled.to_csv('theta.csv', index = False)

np.save('theta.npy', theta_subsampled)

#print(theta_subsampled.shape)


# Save subset of whisker xy deflections that match image timestamps

print("Converting peak protraction xy to .npy")

xy = pd.read_csv("raw_xy.csv")

xy_indices = np.searchsorted(xy['%time'], timestamps['%time']).flatten()

xy_in_sync = xy.iloc[xy_indices].reset_index()

xy_subsampled = xy_in_sync.iloc[:, 3::10]

xy_subsampled.rename(columns = {column: "whisker{}X".format(xy_subsampled.columns.get_loc(column)//2+1) for column in xy_subsampled.columns if not xy_subsampled.columns.get_loc(column) % 2}, inplace = True)
xy_subsampled.rename(columns = {column: "whisker{}Y".format(xy_subsampled.columns.get_loc(column)//2+1) for column in xy_subsampled.columns if xy_subsampled.columns.get_loc(column) % 2}, inplace = True)

xy_subsampled.to_csv('xy.csv', index = False)

#print(xy_subsampled.shape)

np.save('xy.npy', xy_subsampled)


# Save subset of body poses that match image timestamps

print("Converting body pose to .npy")

body_pose = pd.read_csv("raw_whiskeye_body_pose.csv")

body_pose_indices = np.searchsorted(xy['%time'], timestamps['%time']).flatten()

body_pose_in_sync = body_pose.iloc[body_pose_indices].reset_index()

body_pose_subsampled = body_pose_in_sync.iloc[:, 2:]

body_pose_in_sync.rename(columns = {"field.x": "positionX", "field.y": "positionY", "field.theta": "theta"}, inplace = True)

body_pose_subsampled.to_csv('body_pose.csv', index = False)

np.save('body_pose.npy', body_pose_subsampled)


# Save subset of neck orientations that match image timestamps

print("Converting neck pose to .npy")

neck_pose = pd.read_csv("raw_whiskeye_head_neck_meas.csv")

neck_pose_indices = np.searchsorted(xy['%time'], timestamps['%time']).flatten()

neck_pose_in_sync = neck_pose.iloc[neck_pose_indices].reset_index()

neck_pose_subsampled = neck_pose_in_sync.iloc[:, 3:]

neck_pose_in_sync.rename(columns = {"field.data.0": "elevation", "field.data.1": "pitch", "field.data.2": "yaw"}, inplace = True)

neck_pose_subsampled.to_csv('neck_pose.csv', index = False)

np.save('neck_pose.npy', neck_pose_subsampled)


# Process head quaternion pose into (x, y, theta) pose

head_pose = pd.read_csv("ground_truth_head_pose.csv")

position_xy = head_pose.filter(regex = 'position|%time').drop(columns = "field.position.z")

orientation_xyzw = head_pose.filter(regex = 'orientation|%time')

quaternions = Rotation.from_quat(orientation_xyzw[["field.orientation.x", "field.orientation.y", "field.orientation.z", "field.orientation.w"]])

orientation_xyz = quaternions.as_euler('xyz')

yaw_angle_radians = pd.Series(data = orientation_xyz[:,2])

#print(yaw_angle_radians)

ground_truth_x_y_theta = pd.concat([position_xy, yaw_angle_radians], axis = 1)

ground_truth_x_y_theta.rename(columns = {"field.pose.position.x": "position_x", "field.pose.position.y": "position_y", 0: "theta"}, inplace = True)

print(ground_truth_x_y_theta.head())

ground_truth_x_y_theta.to_csv('ground_truth_head_pose_x_y_theta.csv')

np.save('ground_truth_head_pose.npy', ground_truth_x_y_theta.to_numpy())
