import tensorflow as tf
import numpy as np


def max_out(inputs, num_units, axis=None):
  shape = inputs.get_shape().as_list()
  if shape[0] is None:
    shape[0] = -1
  if axis is None:  # Assume that channel is the last dimension
    axis = -1
  num_channels = shape[axis]
  if num_channels % num_units:
    raise ValueError('number of features({}) is not '
                     'a multiple of num_units({})'.format(num_channels,
                                                          num_units))
  shape[axis] = num_units
  shape += [num_channels // num_units]
  outputs = tf.reduce_max(tf.reshape(inputs, shape), -1)
  return outputs


class VanillaCNN(tf.keras.models.Model):
  def __init__(self, hparams, scope="cl_vcnn", *inputs, **kwargs):
    if 'cl_token' in kwargs:
      del kwargs['cl_token']
    super(VanillaCNN, self).__init__(*inputs, **kwargs)
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                                'hc-' + '.'.join(
                                  [str(h) for h in self.hparams.filters]),
                                'hfc-' + '.'.join(
                                  [str(h) for h in self.hparams.fc_dim]),
                                'd-' + str(self.hparams.depth),
                                'hdrop-' + str(
                                  self.hparams.hidden_dropout_rate),
                                'indrop-' + str(
                                  self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.000000002)
    self.create_vars()

  def create_vars(self):

    self.indrop = tf.keras.layers.Dropout(rate=self.hparams.input_dropout_rate)

    self.cnns = []
    self.cnn_nns = []
    self.cnn_bnz = []
    self.cnn_activations = []
    self.cnn_pooling = []
    self.cnn_dropouts = []
    for i in np.arange(self.hparams.depth):
      self.cnns.append(tf.keras.layers.Conv2D(self.hparams.filters[i],
                                              self.hparams.kernel_size[i],
                                              activation=None,
                                              kernel_regularizer=self.regularizer))

      #       if self.hparams.maxout_size[i] < self.hparams.filters[i]:
      #             nn_size = int(self.hparams.filters[i] / self.hparams.maxout_size[i])
      #             self.cnn_nns.append(tf.keras.layers.Conv2D(self.hparams.maxout_size[i],
      #                                                             (1,1),
      #                                       activation=None,
      #                                       kernel_regularizer=self.regularizer))
      #       else:
      #         self.cnn_nns.append(tf.keras.layers.Lambda(lambda x: x))
      self.cnn_bnz.append(tf.keras.layers.BatchNormalization())
      self.cnn_activations.append(tf.keras.layers.Activation('relu'))
      self.cnn_pooling.append(
        tf.keras.layers.MaxPooling2D(self.hparams.pool_size[i]))
      self.cnn_dropouts.append(
        tf.keras.layers.Dropout(rate=self.hparams.hidden_dropout_rate))

    self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()

    self.densez = []
    self.dense_bnz = []
    self.dense_activations = []
    self.dense_dropouts = []

    for i in np.arange(self.hparams.proj_depth):
      self.densez.append(
        tf.keras.layers.Dense(self.hparams.fc_dim[i], activation=None,
                              kernel_regularizer=self.regularizer))
      self.dense_bnz.append(tf.keras.layers.BatchNormalization())
      self.dense_activations.append(tf.keras.layers.Activation('relu'))
      self.dense_dropouts.append(
        tf.keras.layers.Dropout(rate=self.hparams.hidden_dropout_rate))

    self.projector = tf.keras.layers.Dense(self.hparams.output_dim,
                                           kernel_regularizer=self.regularizer)

  def call(self, inputs, padding_symbol=None, training=None, **kwargs):
    x = self.indrop(inputs, training=training, **kwargs)

    for i in np.arange(self.hparams.depth):
      x = self.cnns[i](x, training=training, **kwargs)
      # x = self.cnn_nns[i](x, training=training, **kwargs)
      x = max_out(x, self.hparams.maxout_size[i])
      x = self.cnn_bnz[i](x, training=training, **kwargs)
      x = self.cnn_activations[i](x, training=training, **kwargs)
      x = self.cnn_pooling[i](x, training=training, **kwargs)
      x = self.cnn_dropouts[i](x, training=training, **kwargs)

    x = self.avg_pool(x, **kwargs)

    for i in np.arange(self.hparams.proj_depth):
      x = self.densez[i](x, training=training, **kwargs)
      x = self.dense_bnz[i](x, training=training, **kwargs)
      x = self.dense_activations[i](x, training=training, **kwargs)
      x = self.dense_dropouts[i](x, training=training, **kwargs)

    logits = self.projector(x, training=training, **kwargs)

    return logits

  def detailed_call(self, inputs, padding_symbol=None, **kwargs):
    x = self.indrop(inputs)

    hidden_activations = []
    for i in np.arange(self.hparams.depth):
      x = self.cnns[i](x, **kwargs)
      x = max_out(x, self.hparams.maxout_size[i])
      x = self.cnn_bnz[i](x, **kwargs)
      x = self.cnn_activations[i](x, **kwargs)
      x = self.cnn_pooling[i](x, **kwargs)
      x = self.cnn_dropouts[i](x, **kwargs)
      hidden_activations.append(x)

    x = self.avg_pool(x, **kwargs)
    hidden_activations.append(x)

    for i in np.arange(self.hparams.proj_depth):
      x = self.densez[i](x, **kwargs)
      x = self.dense_bnz[i](x, **kwargs)
      x = self.dense_activations[i](x, **kwargs)
      x = self.dense_dropouts[i](x, **kwargs)
      hidden_activations.append(x)

    logits = self.projector(x, **kwargs)

    return logits, hidden_activations[-1], hidden_activations
