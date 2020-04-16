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
                                'h-' + '.'.join([str(x) for x in self.hparams.hidden_dim]),
                                'd-' + str(self.hparams.depth),
                                'hdrop-' + str(self.hparams.hidden_dropout_rate),
                                'indrop-' + str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.create_vars()
    self.rep_index = 1
    self.rep_layer = -1



  def create_vars(self):
    self.flat = tf.keras.layers.Flatten()
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.batch_norm.trainable = True
    self.indrop = tf.keras.layers.Dropout(self.hparams.input_dropout_rate)

    self.hidden_layers = []
    self.hidden_batch_norms = []
    self.hidden_dropouts = []
    for i in np.arange(self.hparams.depth):
      self.hidden_layers.append(tf.keras.layers.Dense(self.hparams.hidden_dim[i],
                                     activation='relu',
                                     kernel_regularizer=self.regularizer))
      self.hidden_batch_norms.append(tf.keras.layers.BatchNormalization())
      self.hidden_batch_norms[i].trainable = True
      self.hidden_dropouts.append(tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate))

    self.final_dense = tf.keras.layers.Dense(self.hparams.output_dim,
                                   kernel_regularizer=self.regularizer)


  def call(self, inputs, padding_symbol=None, training=None, **kwargs):
    x = self.flat(inputs, **kwargs)
    x = self.batch_norm(x, training=training, **kwargs)
    x = self.indrop(x, training=training, **kwargs)

    for i in np.arange(self.hparams.depth):
      x = self.hidden_layers[i](x, training=training, **kwargs)
      x = self.hidden_batch_norms[i](x, training=training, **kwargs)
      x = self.hidden_dropouts[i](x, training=training, **kwargs)

    logits = self.final_dense(x, training=training, **kwargs)

    return logits


  def detailed_call(self, inputs, padding_symbol=None, training=None, **kwargs):
    layer_activations = []
    x = self.flat(inputs, **kwargs)
    x = self.batch_norm(x, training=training, **kwargs)
    x = self.indrop(x, training=None, **kwargs)
    layer_activations.append(x)

    for i in np.arange(self.hparams.depth):
      x = self.hidden_layers[i](x, training=None, **kwargs)
      x = self.hidden_batch_norms[i](x, training=None, **kwargs)
      x = self.hidden_dropouts[i](x, training=None, **kwargs)
      layer_activations.append(x)

    pnltimt = x
    logits = self.final_dense(x, training=None, **kwargs)

    return logits, pnltimt, layer_activations

