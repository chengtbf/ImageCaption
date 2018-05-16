# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base

slim = tf.contrib.slim

def inception_v3(images,
                 trainable=False,
                 is_training=False,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
          shape = net.get_shape()
          net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
          '''
          net = slim.dropout(
              net,
              keep_prob=dropout_keep_prob,
              is_training=is_inception_model_training,
              scope="dropout")
          '''
          net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net

train_list_file = "data/train_list.txt"
valid_list_file = "data/valid_list.txt"
test_list_file = "data/test_list.txt"
train_list_file = open(train_list_file, 'r')
valid_list_file = open(valid_list_file, 'r')
test_list_file = open(test_list_file, 'r')
train_image_list = []
test_image_list = []
valid_image_list = []

for line in train_list_file.readlines():
    train_image_list.append(line.strip().split()[0])


for line in valid_list_file.readlines():
    valid_image_list.append(line.strip().split()[0])


for line in test_list_file.readlines():
    test_image_list.append(line.strip().split()[0])


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
image_root_path = "/home/chengcheng/dataset/image_caption/image/Flicker8k_Dataset/"
image_list = os.listdir(image_root_path)
image_path_list = []
for image_file_name in image_list:
    image_path_list.append(image_root_path + image_file_name)
# image_path_list = image_path_list[0:10]

# print(image_path_list)
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(image_path_list)
#            tf.train.match_filenames_once(image_root_path + "*.jpg"))

# Read an entire image file which is required since they're JPEGs, if the images
# are too large they could be split in advance to smaller files or use the Fixed
# reader to split up the file.
image_reader = tf.WholeFileReader()

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
image_file_name, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.image.resize_images(image, size=[299,299],method=tf.image.ResizeMethod.BILINEAR)
# image = tf.image.resize_image_with_crop_or_pad(image, 299, 299)
image = tf.subtract(image, 0.5)
image = tf.multiply(image, 2.0)
image = tf.reshape(image, [1, 299, 299, 3])
inception_output = inception_v3(image)
inception_output = tf.reshape(inception_output, [2048])
saver = tf.train.Saver()

image2feat = {}

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    # tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver.restore(sess, "/home/chengcheng/dataset/image_caption/model/inception_v3.ckpt")
    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Get an image tensor and print its value.
    # image_tensor = sess.run([image])
    # print(image_tensor)
    for i in range(len(image_list)):
    # for i in range(10):
        key, value= sess.run([image_file_name ,inception_output])
        key = key.split('/')[-1]
        # print(key, value)
        image2feat[key] = value
        if i % 100 == 0:
            print(i,' Done')
        '''
        value = inception_output.eval()
        print(value)
        image_tensor = image.eval()
        image_file_name_str = image_file_name.eval()
        key = image_file_name_str.split('/')[-1]
        print(key)
        '''
        # print(image)
        # print(image_tensor.shape)
        # print(image_tensor)
        # print(image_file_name.eval())

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

import h5py
import numpy as np

f = h5py.File('/home/chengcheng/dataset/image_caption/feat2.hdf5', 'w')
# train_set = f.ceate_dataset("train_set", ())

train_feat_list = []
for train_image in train_image_list:
    if train_image not in image2feat:
        print("train image: " + train_image + " not found!")
        continue
    train_feat_list.append(image2feat[train_image])
    # print("train ",train_image)
train_set = f.create_dataset("train_set", data=np.array(train_feat_list))
print("train_set done.")

valid_feat_list = []
for valid_image in valid_image_list:
    if valid_image not in image2feat:
        print("valid_image:" + valid_image + " not found!")
        continue
    valid_feat_list.append(image2feat[valid_image])
    # print("valid ",valid_image)
valid_set = f.create_dataset("valid_set", data=np.array(valid_feat_list))
print("valid_set done.")

test_feat_list = []
for test_image in test_image_list:
    if test_image not in image2feat:
        print("test_image: " + test_image + " not found!")
        continue
    test_feat_list.append(image2feat[test_image])
    # print("test ",test_image)
test_set = f.create_dataset("test_set", data=np.array(test_feat_list))
print("test_set done.")

f.close()
