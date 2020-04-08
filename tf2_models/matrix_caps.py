import tensorflow as tf
from tf2_models.caps_util import *
from tf2_models.caps_layers import FcCaps, ConvCaps


class MatrixCaps(tf.keras.Model):
  def __init__(self, hparams, scope='matrix_caps', *inputs, **kwargs):
    super(MatrixCaps, self).__init__(hparams, name=scope, *inputs, **kwargs)

    self.hparams = hparams

    # xavier initialization is necessary here to provide higher stability
    # instead of initializing bias with constant 0, a truncated normal
    # initializer is exploited here for higher stability
    self.bias_initializer = tf.keras.initializers.glorot_normal()

    # "We use a weight decay loss with a small factor of .0000002 rather than
    # the reconstruction loss."
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJeQnSsE3X
    self.weights_regularizer = tf.keras.regularizers.l2(self.hparams.l2)

    self.batch_norm = tf.keras.layers.BatchNormalization()

    self.conv1 = tf.keras.layers.Conv2D(filters=self.hparams.A, kernel_size=[5,5], strides=(2, 2),
                                        padding='same',
                                        activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer='glorot_uniform',
                                        bias_initializer=self.bias_initializer,
                                        kernel_regularizer=self.weights_regularizer,
                                        bias_regularizer=self.weights_regularizer,
                                        activity_regularizer=None,
                                        kernel_constraint=None,
                                        bias_constraint=None)
    self.primcaps_pose_conv = tf.keras.layers.Conv2D(filters=self.hparams.B * 16, kernel_size=[1,1], strides=(1, 1),
                               padding='valid',
                               activation=None, use_bias=True,
                               kernel_initializer='glorot_uniform',
                               bias_initializer='glorot_uniform',
                               kernel_regularizer=self.weights_regularizer,
                               bias_regularizer=self.weights_regularizer,
                               activity_regularizer=None,
                               kernel_constraint=None, bias_constraint=None)
    self.primcaps_activation_conv = tf.keras.layers.Conv2D(filters=self.hparams.B, kernel_size=[1,1], strides=(1, 1),
                               padding='valid',
                               activation=tf.nn.sigmoid, use_bias=True,
                               kernel_initializer='glorot_uniform',
                               bias_initializer='glorot_uniform',
                               kernel_regularizer=self.weights_regularizer,
                               bias_regularizer=self.weights_regularizer,
                               activity_regularizer=None,
                               kernel_constraint=None, bias_constraint=None)

    self.convcaps1 = ConvCaps(self.hparams,
                              num_output_caps=self.hparams.C,
                              kernel=3, stride=2, kh_kw_i=9*8,
                              scope='conv_caps1')
    self.convcaps2 = ConvCaps(self.hparams,
                              num_output_caps=self.hparams.D,
                              kernel=3, stride=1, kh_kw_i=9*16,
                              scope='conv_caps2')


    self.fc_caps = FcCaps(self.hparams,
                          scope='class_caps')


  @tf.function
  def call(self, inputs, padding_symbol=None, training=True, **kwargs):
    inputs_shapes = tf.shape(inputs)
    batch_size = inputs_shapes[0]
    spatial_size = inputs_shapes[1]
    tf.print(batch_size, spatial_size)
    outputs = self.batch_norm(inputs)
    outputs = self.conv1(outputs)
    outputs_pos = self.primcaps_pose_conv(outputs)
    outputs_activations = self.primcaps_activation_conv(outputs)

    spatial_size = tf.shape(outputs_pos)[1]
    outputs_pos = tf.reshape(outputs_pos, shape=[batch_size, spatial_size, spatial_size,
                                   self.hparams.B, 16], name='pose')

    outputs_activations = tf.reshape(
      outputs_activations,
      shape=[batch_size, spatial_size, spatial_size, self.hparams.B, 1],
      name="activation")

    # assert [a for a in tf.shape(outputs_pos)] == [batch_size, spatial_size, spatial_size,
    #                             self.hparams.B, 16]
    # assert [a for a in tf.shape(outputs_activations)] == [batch_size, spatial_size, spatial_size,
    #                                   self.hparams.B, 1]

    # ----- Conv Caps 1 -----#
    # activation_in: (64, 7, 7, 8, 1)
    # pose_in: (64, 7, 7, 16, 16)
    # activation_out: (64, 5, 5, 32, 1)
    # pose_out: (64, 5, 5, 32, 16)
    tf.print("canvcaps1 inputs", outputs_pos.shape, outputs_activations.shape)
    outputs_pos, outputs_activations = self.convcaps1(outputs_pos, outputs_activations, training, **kwargs)
    tf.print('convcaps1 outputs_pos', outputs_pos.shape)
    tf.print('convcaps1 outputs_activations', outputs_activations.shape)
    # ----- Conv Caps 2 -----#
    # activation_in: (64, 7, 7, 8, 1)
    # pose_in: (64, 7, 7, 16, 1)
    # activation_out: (64, 5, 5, 32, 1)
    # pose_out: (64, 5, 5, 32, 16)
    outputs_pos, outputs_activations = self.convcaps2(outputs_pos, outputs_activations, training, **kwargs)
    tf.print('convcaps2 outputs_pos', outputs_pos.shape)
    tf.print('convcaps2 outputs_activations', outputs_activations.shape)
    # ----- Class Caps -----#
    # activation_in: (64, 5, 5, 32, 1)
    # pose_in: (64, 5, 5, 32, 16)
    # activation_out: (64, 5)
    # pose_out: (64, 5, 16)
    outputs_pos, outputs_activations = self.fc_caps(outputs_pos, outputs_activations, training, **kwargs)

    tf.print('outputs_pos', outputs_pos.shape)
    tf.print('outputs_activations', outputs_activations.shape)

    return outputs_activations

