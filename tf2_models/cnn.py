import tensorflow as tf
import numpy as np

class VanillaCNN(tf.keras.models.Sequential):
  def __init__(self, hparams, scope="cl_vcnn", *inputs, **kwargs):
    if 'cl_token' in kwargs:
      del kwargs['cl_token']
    super(VanillaCNN, self).__init__(*inputs, **kwargs)
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                                'h-' + str(self.hparams.hidden_dim),
                                'd-' + str(self.hparams.depth),
                                'hdrop-' + str(self.hparams.hidden_dropout_rate),
                                'indrop-' + str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.create_vars()

  def create_vars(self):
    self.add(tf.keras.layers.ZeroPadding2D((2,2)))

    for i in np.arange(self.hparams.depth):
      self.add(tf.keras.layers.Conv2D(self.hparams.filters[i], self.hparams.kernel_size[i], activation='relu',
                                      input_shape=(32, 32, 1)))
      self.add(tf.keras.layers.MaxPooling2D(self.hparams.pool_size[i]))

    self.add(tf.keras.layers.Flatten())

    for i in np.arange(self.hparams.proj_depth):
      self.add(tf.keras.layers.Dense(self.hparams.hidden_dim[i], activation='relu'))

    self.add(tf.keras.layers.Dense(self.hparams.output_dim))
