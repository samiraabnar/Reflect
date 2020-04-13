import tensorflow as tf
import numpy as np


class VanillaFF(tf.keras.models.Sequential):
  def __init__(self, hparams, scope="cl_vff", *inputs, **kwargs):
    if 'cl_token' in kwargs:
      del kwargs['cl_token']

    super(VanillaFF, self).__init__()
    self.scope = scope
    self.hparams = hparams

    self.model_name = '_'.join([self.scope,
                                'h-' + str(self.hparams.hidden_dim),
                                'd-' + str(self.hparams.depth),
                                'hdrop-' + str(self.hparams.hidden_dropout_rate),
                                'indrop-' + str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.create_vars()


  def create_vars(self):
    self.add(tf.keras.layers.Flatten())
    self.add(tf.keras.layers.BatchNormalization())
    self.add(tf.keras.layers.Dropout(self.hparams.input_dropout_rate))

    for i in np.arange(self.hparams.depth):
      self.add(tf.keras.layers.Dense(self.hparams.hidden_dim, activation='relu'))
      self.add(tf.keras.layers.BatchNormalization())
      self.add(tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate))

    self.add(tf.keras.layers.Dense(self.hparams.output_dim))

