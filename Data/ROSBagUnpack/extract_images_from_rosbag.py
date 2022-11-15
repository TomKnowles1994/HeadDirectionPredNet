import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import os
from glob import iglob
from argparse import ArgumentParser

import rosbag
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

preserve_pose_index = True          # True if you want to keep the original pose index as a column in the final dataset. False if you want MultiPredNet-ready data.
preserve_pose_timestamps = True     # True if you want to keep the pose timestamps in the final dataset. False if you want MultiPredNet-ready data.

parser = ArgumentParser()
parser.add_argument('bagfile')

bag_file = parser.parse_args().bagfile

jpg_quality = 100

compr = cv.IMWRITE_JPEG_QUALITY;

quality = jpg_quality  # jpg quality is in [0,100] range, png [0,9]

params = [compr, quality]

bag = rosbag.Bag(bag_file, "r")

bridge = CvBridge()


# Image topic cam0

timestamps = pd.read_csv("raw_images_cam0.csv", usecols = ['%time'])

output_dir = "cam0/"

image_topic = "/whiskeye/head/cam0/image_raw/compressed"

count = 0

for topic, msg, t in bag.read_messages(topics=[image_topic]):

	cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")

	save_name = "{}.jpg".format(count)

	save_path = os.path.join(output_dir, save_name)

	cv.imwrite(save_path, cv_img, params)

	count += 1

bag.close()
