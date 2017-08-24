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
"""Rgb VAE - GAN implementation for CloudML."""

import argparse
import os

import tensorflow as tf

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

import util
from util import override_if_not_in_args

# Global constants for Rgb dataset
EMBEDDING_DIMENSION = 100
LAYER_DIM = 64
TRAIN, EVAL = 'TRAIN', 'EVAL'
PREDICT_EMBED_IN, PREDICT_IMAGE_IN = 'PREDICT_EMBED_IN', 'PREDICT_IMAGE_IN'


def build_signature(inputs, outputs):
  """Build the signature for use when exporting the graph.

  Args:
    inputs: a dictionary from tensor name to tensor
    outputs: a dictionary from tensor name to tensor
  Returns:
    The signature, a SignatureDef proto, specifies the input/output tensors
    to bind when running prediction.
  """
  signature_inputs = {
      key: saved_model_utils.build_tensor_info(tensor)
      for key, tensor in inputs.items()
  }
  signature_outputs = {
      key: saved_model_utils.build_tensor_info(tensor)
      for key, tensor in outputs.items()
  }

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type=float, default=0.0002)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument('--beta1', type=float, default=0.5)
  parser.add_argument('--resized_image_size', type=int, default=64)
  parser.add_argument('--crop_image_dimension', type=int, default=None)
  parser.add_argument(
      '--center_crop', dest='center_crop', default=False, action='store_true')
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '80000', task_args)
  override_if_not_in_args('--batch_size', '64', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)

  return Model(args.learning_rate, args.dropout, args.beta1,
               args.resized_image_size, args.crop_image_dimension,
               args.center_crop), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.image = None
    self.label = None
    self.global_step = None
    self.keys = None
    self.predictions = None
    self.embeddings = []
    self.cost_encoder = None
    self.cost_generator = None
    self.cost_discriminator = None
    self.cost_balance = None
    self.prediction_image = None
    self.dis_real = None
    self.dis_fake = None
    self.encoder_optimizer = None
    self.generator_optimizer = None
    self.discriminator_optimizer = None


class Model(object):
  """Tensorflow model for Rgb VAE-GAN."""

  def __init__(self, learning_rate, dropout, beta1, resized_image_size,
               crop_image_dimension, center_crop):
    """Initializes VAE-GAN. DCGAN architecture: https://arxiv.org/abs/1511.06434

    Args:
      learning_rate: The learning rate for the three networks.
      dropout: The dropout rate for training the network.
      beta1: Exponential decay rate for the 1st moment estimates.
      resized_image_size: Desired size of resized image.
      crop_image_dimension: Square size of the bounding box.
      center_crop: True iff images should be center cropped.
    """
    self.learning_rate = learning_rate
    self.dropout = dropout
    self.beta1 = beta1
    self.resized_image_size = resized_image_size
    self.crop_image_dimension = crop_image_dimension
    self.center_crop = center_crop
    self.has_exported_embed_in = False
    self.has_exported_image_in = False
    self.batch_size = 0

  def leaky_relu(self, x, name, leak=0.2):
    """Leaky relu activation function.

    Args:
      x: input into layer.
      name: name scope of layer.
      leak: slope that provides non-zero y when x < 0.

    Returns:
      The leaky relu activation.
    """
    return tf.maximum(x, leak * x, name=name)

  def build_graph(self, data_dir, batch_size, mode):
    """Builds the VAE-GAN network.

    Args:
      data_dir: Locations of input data.
      batch_size: Batch size of input data.
      mode: Mode of the graph (TRAINING, EVAL, or PREDICT)

    Returns:
      The tensors used in training the model.
    """
    tensors = GraphReferences()
    assert batch_size > 0
    self.batch_size = batch_size
    if mode is PREDICT_EMBED_IN:
      # Input embeddings to send through decoder/generator network.
      tensors.embeddings = tf.placeholder(
          tf.float32, shape=(None, EMBEDDING_DIMENSION), name='input')
    elif mode is PREDICT_IMAGE_IN:
      tensors.prediction_image = tf.placeholder(
          tf.string, shape=(None,), name='input')
      tensors.image = tf.map_fn(
          self.process_image, tensors.prediction_image, dtype=tf.float32)

    if mode in (TRAIN, EVAL):
      mode_string = 'train'
      if mode is EVAL:
        mode_string = 'validation'

      tensors.image = util.read_and_decode(
          data_dir, batch_size, mode_string, self.resized_image_size,
          self.crop_image_dimension, self.center_crop)

      tensors.image = tf.reshape(tensors.image, [
          -1, self.resized_image_size, self.resized_image_size, 3
      ])

      tf.summary.image('original_images', tensors.image, 1)

      tensors.embeddings, y_mean, y_stddev = self.encode(tensors.image)

    if mode is PREDICT_IMAGE_IN:
      tensors.image = tf.reshape(tensors.image, [
          -1, self.resized_image_size, self.resized_image_size, 3
      ])
      tensors.embeddings, y_mean, _ = self.encode(tensors.image, False)
      tensors.predictions = tensors.embeddings
      return tensors

    decoded_images = self.decode(tensors.embeddings)

    if mode is TRAIN:
      tf.summary.image('decoded_images', decoded_images, 1)

    if mode is PREDICT_EMBED_IN:
      decoded_images = self.decode(tensors.embeddings, False, True)
      output_images = (decoded_images + 1.0) / 2.0
      output_img = tf.image.convert_image_dtype(
          output_images, dtype=tf.uint8, saturate=True)[0]
      output_data = tf.image.encode_png(output_img)
      output = tf.encode_base64(output_data)

      tensors.predictions = output

      return tensors

    tensors.dis_fake = self.discriminate(decoded_images, self.dropout)
    tensors.dis_real = self.discriminate(
        tensors.image, self.dropout, reuse=True)

    tensors.cost_encoder = self.loss_encoder(tensors.image, decoded_images,
                                             y_mean, y_stddev)
    tensors.cost_generator = self.loss_generator(tensors.dis_fake)
    tensors.cost_discriminator = self.loss_discriminator(
        tensors.dis_real, tensors.dis_fake)

    if mode in (TRAIN, EVAL):
      tf.summary.scalar('cost_encoder', tensors.cost_encoder)
      tf.summary.scalar('cost_generator', tensors.cost_generator)
      tf.summary.scalar('cost_discriminator', tensors.cost_discriminator)
      tf.summary.tensor_summary('disc_fake', tensors.dis_fake)
      tf.summary.tensor_summary('disc_real', tensors.dis_real)
      tf.summary.scalar('mean_disc_fake', tf.reduce_mean(tensors.dis_fake))
      tf.summary.scalar('mean_disc_real', tf.reduce_mean(tensors.dis_real))

    # Cost of Decoder/Generator is VAE network cost and cost of generator
    # being detected by the discriminator.
    enc_weight = 1
    gen_weight = 1
    tensors.cost_balance = (
        enc_weight * tensors.cost_encoder + gen_weight * tensors.cost_generator)

    tensors.global_step = tf.Variable(0, name='global_step', trainable=False)
    t_vars = tf.trainable_variables()

    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
      encoder_vars = [var for var in t_vars if var.name.startswith('enc_')]
      generator_vars = [var for var in t_vars if var.name.startswith('gen_')]
      discriminator_vars = [
          var for var in t_vars if var.name.startswith('disc_')
      ]
      vae_vars = encoder_vars + generator_vars

      # Create optimizers for each network.
      tensors.encoder_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate, beta1=self.beta1).minimize(
              tensors.cost_encoder,
              var_list=vae_vars,
              global_step=tensors.global_step)
      tensors.generator_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate, beta1=self.beta1).minimize(
              tensors.cost_balance,
              var_list=vae_vars,
              global_step=tensors.global_step)
      tensors.discriminator_optimizer = tf.train.AdamOptimizer(
          learning_rate=self.learning_rate, beta1=self.beta1).minimize(
              tensors.cost_discriminator,
              var_list=discriminator_vars,
              global_step=tensors.global_step)

    return tensors

  def build_train_graph(self, data_paths, batch_size):
    """Builds the training VAE-GAN graph.

    Args:
      data_paths: Locations of input data.
      batch_size: Batch size of input data.

    Returns:
      The tensors used in training the model.
    """
    return self.build_graph(data_paths, batch_size, mode=TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    """Builds the evaluation VAE-GAN graph.

    Args:
      data_paths: Locations of input data.
      batch_size: Batch size of input data.

    Returns:
      The tensors used in training the model.
    """
    return self.build_graph(data_paths, batch_size, mode=EVAL)

  def build_prediction_embedding_graph(self):
    """Builds the prediction VAE-GAN graph for embedding input.

    Returns:
      The inputs and outputs of the prediction.
    """
    tensors = self.build_graph(None, 1, PREDICT_EMBED_IN)

    keys_p = tf.placeholder(tf.string, shape=[None])
    inputs = {'key': keys_p, 'embeddings': tensors.embeddings}
    keys = tf.identity(keys_p)
    outputs = {'key': keys, 'prediction': tensors.predictions}

    return inputs, outputs

  def build_prediction_image_graph(self):
    """Builds the prediction VAE-GAN graph for image input.

    Returns:
      The inputs and outputs of the prediction.
    """
    tensors = self.build_graph(None, 1, PREDICT_IMAGE_IN)

    keys_p = tf.placeholder(tf.string, shape=[None])
    inputs = {'key': keys_p, 'image_bytes': tensors.prediction_image}
    keys = tf.identity(keys_p)
    outputs = {'key': keys, 'prediction': tensors.predictions}

    return inputs, outputs

  def encode(self, images, is_training=True, reuse=None):
    """Encoder network for VAE.

    Args:
      images: Images to encode to latent space vector.
      is_training: True iff in training mode.
      reuse: True iff variables should be reused.

    Returns:
      The embedding vector, mean and standard deviation vectors.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=None):
      # Convolution Layer 1
      conv = self.leaky_relu(
          tf.layers.conv2d(
              inputs=images,
              filters=LAYER_DIM,
              kernel_size=4,
              strides=(2, 2),
              padding='same',
              name='enc_conv0'), 'enc_r0')

      layers = [conv]
      for i, filters in enumerate([LAYER_DIM * 2, LAYER_DIM * 4,
                                   LAYER_DIM * 8]):
        # Convolutional Layer
        conv = tf.layers.conv2d(
            inputs=layers[-1],
            filters=filters,
            kernel_size=4,
            strides=(2, 2),
            padding='same',
            name='enc_conv' + str(i + 1))

        # Batch Norm Layer
        bn = tf.contrib.layers.batch_norm(
            conv,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            reuse=reuse,
            scope='enc_bn' + str(i + 1),
            is_training=is_training)

        # ReLU activation
        relu = self.leaky_relu(bn, name='enc_rl' + str(i + 1))
        layers.append(relu)

      # Fully Connected Layer
      conv4_flat = tf.reshape(layers[-1], [-1, 4 * 4 * LAYER_DIM * 8])

      # Get Mean and Standard Deviation Vectors
      y_mean = tf.layers.dense(
          inputs=conv4_flat,
          units=EMBEDDING_DIMENSION,
          activation=None,
          name='enc_y_mean')
      y_stddev = tf.layers.dense(
          inputs=conv4_flat,
          units=EMBEDDING_DIMENSION,
          activation=None,
          name='enc_y_stddev')
      samples = tf.random_normal(
          [self.batch_size, EMBEDDING_DIMENSION], 0, 1, dtype=tf.float32)

      y_vector = y_mean + (y_stddev * samples)
      return y_vector, y_mean, y_stddev

  def decode(self, embeddings, is_training=True, reuse=False):
    """Decoder network for VAE / Generator network for GAN.

    Args:
      embeddings: Vector to decode into images.
      is_training: True iff in training mode.
      reuse: True iff vars should be reused.

    Returns:
      The decoded images.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      # Fully Connected Layers
      fc3 = tf.layers.dense(
          inputs=embeddings,
          units=4 * 4 * LAYER_DIM * 8,
          activation=None,
          name='gen_fc3')
      fc3_reshaped = tf.reshape(fc3, [-1, 4, 4, LAYER_DIM * 8])

      layers = [fc3_reshaped]
      for i, filters in enumerate([LAYER_DIM * 4, LAYER_DIM * 2, LAYER_DIM]):
        # Batch Norm Layer
        bn = tf.contrib.layers.batch_norm(
            layers[-1],
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            reuse=reuse,
            scope='gen_bn' + str(i),
            is_training=is_training)

        # ReLU activation
        relu = tf.nn.relu(bn, name='gen_rl' + str(i))

        # "Deconvolution" Layer
        deconv = tf.layers.conv2d_transpose(
            inputs=relu,
            filters=filters,
            kernel_size=4,
            strides=(2, 2),
            padding='same',
            name='gen_deconv' + str(i))
        layers.append(deconv)

      # Batch norm
      bn = tf.nn.relu(
          tf.contrib.layers.batch_norm(
              layers[-1],
              decay=0.9,
              updates_collections=None,
              epsilon=1e-5,
              scale=True,
              reuse=None,
              scope='gen_bn3',
              is_training=is_training),
          name='gen_rl3')

      # "Deconvolution" Layer 3
      deconv = tf.layers.conv2d_transpose(
          inputs=bn,
          filters=3,
          kernel_size=4,
          strides=(2, 2),
          padding='same',
          activation=tf.nn.tanh,
          name='gen_deconv3')

      return deconv

  def discriminate(self, input_images, dropout=0.5, reuse=False):
    """Decoder network for VAE / Generator network for GAN.

    Args:
      input_images: Input images to discriminate.
      dropout: Dropout used for training.
      reuse: True iff variables should be in reuse mode.

    Returns:
      Whether the images are real or fake.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
      # Convolution Layer 1
      conv = self.leaky_relu(
          tf.layers.conv2d(
              inputs=input_images,
              filters=LAYER_DIM,
              kernel_size=4,
              strides=(2, 2),
              padding='same',
              name='disc_conv0'), 'disc_r0')

      layers = [conv]
      for i, filters in enumerate([LAYER_DIM * 2, LAYER_DIM * 4,
                                   LAYER_DIM * 8]):
        # Convolutional Layer
        conv = tf.layers.conv2d(
            inputs=layers[-1],
            filters=filters,
            kernel_size=4,
            strides=(2, 2),
            padding='same',
            name='disc_conv' + str(i + 1))

        # Batch Norm Layer
        bn = tf.contrib.layers.batch_norm(
            conv,
            decay=0.9,
            updates_collections=None,
            epsilon=1e-5,
            scale=True,
            reuse=reuse,
            scope='disc_bn' + str(i + 1),
            is_training=True)

        # ReLU activation
        relu = self.leaky_relu(bn, name='disc_rl' + str(i + 1))
        layers.append(relu)

      # Fully Connected Layer
      conv_flat = tf.reshape(layers[-1], [-1, 4 * 4 * LAYER_DIM * 8])
      dropout_output = tf.nn.dropout(conv_flat, dropout)
      fc = tf.layers.dense(
          inputs=dropout_output,
          units=1,
          activation=tf.nn.sigmoid,
          name='disc_fc0')

      return fc

  def loss_encoder(self, images, d_images, mean, stddev):
    """Computes the loss of the VAE.

    Args:
      images: The input images to the VAE.
      d_images: The decoded images produced by the VAE.
      mean: The mean vector output by the encoder.
      stddev: The sttdev vector output by the encoder.

    Returns:
      The cost of the VAE.
    """
    cost_reconstruct = tf.reduce_sum(tf.square(images - d_images))

    cost_latent = 0.5 * tf.reduce_sum(
        tf.square(mean) + tf.square(stddev) -
        tf.log(tf.maximum(tf.square(stddev), 1e-10)) - 1, 1)

    cost_encoder = tf.reduce_mean(cost_latent + cost_reconstruct)
    return cost_encoder / (self.resized_image_size * self.resized_image_size)

  def loss_generator(self, dis_fake):
    """Computes the loss of the generator network.

    Args:
      dis_fake: The output of the discriminator for the fake images.

    Returns:
      The cost of the generator.
    """
    return tf.reduce_mean(-1 * tf.log(tf.clip_by_value(dis_fake, 1e-10, 1.0)))

  def loss_discriminator(self, dis_real, dis_fake):
    """Computes the loss of the discriminator network.

    Args:
      dis_real: The output of the discriminator for the real images.
      dis_fake: The output of the discriminator for the fake images.

    Returns:
      The cost of the discriminator.
    """
    return tf.reduce_mean(-1 *
                          (tf.log(tf.clip_by_value(dis_real, 1e-10, 1.0)) +
                           tf.log(tf.clip_by_value(1 - dis_fake, 1e-10, 1.0))))

  def export(self, last_checkpoint, output_dir):
    """Exports the prediction graph.

    Args:
      last_checkpoint: The last checkpoint saved.
      output_dir: Directory to save graph.
    """
    if not self.has_exported_embed_in:
      with tf.Session(graph=tf.Graph()) as sess:
        inputs, outputs = self.build_prediction_embedding_graph()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        trained_saver = tf.train.Saver()
        trained_saver.restore(sess, last_checkpoint)

        predict_signature_def = build_signature(inputs, outputs)
        # Create a saver for writing SavedModel training checkpoints.
        build = builder.SavedModelBuilder(
            os.path.join(output_dir, 'saved_model_embed_in'))
        build.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
        self.has_exported_embed_in = True
        build.save()

    if not self.has_exported_image_in:
      with tf.Session(graph=tf.Graph()) as sess:
        inputs, outputs = self.build_prediction_image_graph()
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        trained_saver = tf.train.Saver()
        trained_saver.restore(sess, last_checkpoint)

        predict_signature_def = build_signature(inputs, outputs)
        # Create a saver for writing SavedModel training checkpoints.
        build = builder.SavedModelBuilder(
            os.path.join(output_dir, 'saved_model_image_in'))
        build.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    predict_signature_def
            },
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))
        self.has_exported_image_in = True
        build.save()

  def process_image(self, input_img):
    image = tf.image.decode_jpeg(input_img, channels=3)
    image = tf.image.central_crop(image, 0.75)
    image = tf.image.resize_images(
        image, [self.resized_image_size, self.resized_image_size])
    image.set_shape((self.resized_image_size, self.resized_image_size, 3))

    image = tf.cast(image, tf.float32) / 127.5 - 1
    return image

