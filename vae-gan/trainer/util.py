# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utility file for VAE-GAN."""

import os
import tensorflow as tf


def read_and_decode(data_directory, batch_size, mode, resized_image_size,
                    crop_image_size, center_crop):
  """Reads and decodes TF Record files.

  Based on example for reading/decoding MNIST digits:
  https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py

  Args:
    data_directory: directory containing tfrecord files
    batch_size: Batch size for input data.
    mode: defines Train or Validation modes.
    resized_image_size: Desired size of image.
    crop_image_size: Original size to crop image.
    center_crop: True iff image should be center cropped.

  Returns:
    Batch of input images to train/validate model.
  """
  tf_record_pattern = os.path.join(data_directory[0], '%s-*' % mode)
  data_files = tf.gfile.Glob(tf_record_pattern)

  queue = tf.train.string_input_producer(data_files)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(queue)

  features = tf.parse_single_example(
      serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/height': tf.FixedLenFeature([], tf.int64),
          'image/width': tf.FixedLenFeature([], tf.int64),
      })

  image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  original_image_height = tf.cast(features['image/height'], tf.int32)
  original_image_width = tf.cast(features['image/width'], tf.int32)

  if crop_image_size is None:
    crop_image_size = tf.cast(
        tf.minimum(original_image_width, original_image_height), tf.float32)

  # Crop rectangular image to centered bounding box.
  assert original_image_height > crop_image_size
  assert original_image_width > crop_image_size
  if center_crop:
    image = tf.image.crop_to_bounding_box(
        image, (original_image_height - crop_image_size) / 2,
        (original_image_width - crop_image_size) / 2, crop_image_size,
        crop_image_size)
  else:
    image = tf.image.crop_to_bounding_box(
        image,
        tf.random_uniform(
            [],
            dtype=tf.int32,
            maxval=(original_image_height - crop_image_size)),
        tf.random_uniform(
            [], dtype=tf.int32,
            maxval=(original_image_width - crop_image_size)), crop_image_size,
        crop_image_size)

  # Resize image to desired pixel dimensions.
  image = tf.image.resize_images(image,
                                 [resized_image_size, resized_image_size])
  image.set_shape((resized_image_size, resized_image_size, 3))

  image = tf.cast(image, tf.float32) * (1. / 127.5) - 1
  images = tf.train.shuffle_batch(
      [image],
      batch_size=batch_size,
      num_threads=1,
      capacity=1000 + 3 * batch_size,
      min_after_dequeue=1000)

  return images


def override_if_not_in_args(flag, argument, args):
  """Checks if flags is in args, and if not it adds the flag to args."""
  if flag not in args:
    args.extend([flag, argument])
