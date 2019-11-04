import absl
import tensorflow as tf
import numpy as np
from tensorboard.compat.tensorflow_stub import tensor_shape
from tensorflow.python.util import nest

from tf2_models.common_layers import get_initializer
from tf2_models.embedding import SharedEmbeddings
from tf2_models.utils import create_init_var


class LmLSTM(tf.keras.Model):

  def __init__(self, hparams, scope="lm_lstm",*inputs, **kwargs):
    del kwargs['cl_token']
    super(LmLSTM, self).__init__(*inputs, **kwargs)
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'em-'+str(self.hparams.embedding_dim),
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.create_vars()

  @tf.function
  def create_vars(self):
    self.input_embedding = tf.compat.v2.keras.layers.Embedding(input_dim=self.hparams.input_dim,
                                                               output_dim=self.hparams.embedding_dim,
                                                               input_shape=(None, None),
                                                               mask_zero=True,
                                                               embeddings_regularizer=self.regularizer,
                                                               name='input_embedding')
    self.input_embedding_dropout = tf.keras.layers.Dropout(self.hparams.input_dropout_rate)
    self.output_embedding_dropout = tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate)

    self.output_embedding = tf.compat.v2.keras.layers.Dense(units=self.hparams.output_dim,
                                                            kernel_regularizer=self.regularizer,
                                                            bias_regularizer=self.regularizer,
                                                            name='output_projection')

    self.stacked_rnns = []
    for _ in np.arange(self.hparams.depth):
      self.stacked_rnns.append(tf.keras.layers.LSTM(units=self.hparams.hidden_dim,
                                                    return_sequences=True,
                                                    return_state=True,
                                                    go_backwards=False,
                                                    stateful=False,
                                                    unroll=False,
                                                    time_major=False,
                                                    recurrent_dropout=self.hparams.hidden_dropout_rate,
                                                    dropout=self.hparams.hidden_dropout_rate,
                                                    kernel_regularizer=self.regularizer,
                                                    recurrent_regularizer=self.regularizer,
                                                    bias_regularizer=self.regularizer,

                                                    ))

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, **kwargs):
    if 'training' in kwargs:
      training = kwargs['training']
    else:
      training = False

    embedded_input = self.input_embedding_dropout(self.input_embedding(inputs),training=training)
    rnn_outputs = embedded_input

    input_mask = self.input_embedding.compute_mask(inputs)
    float_input_mask = tf.cast(input_mask, dtype=tf.float32)
    for i in np.arange(self.hparams.depth):
      rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs, mask=input_mask, training=training)

    rnn_outputs = self.output_embedding_dropout(rnn_outputs, training=training)
    logits = self.output_embedding(rnn_outputs)
    logits = logits * float_input_mask[...,None] + tf.eye(self.hparams.output_dim)[0] * (1 - float_input_mask[...,None])

    return logits


class ClassifierLSTM(tf.keras.Model):

  def __init__(self, hparams, scope="cl_lstm", *inputs, **kwargs):
    del kwargs['cl_token']
    super(ClassifierLSTM, self).__init__(*inputs, **kwargs)
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'em-'+str(self.hparams.embedding_dim),
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00001)
    self.create_vars()

  @tf.function
  def create_vars(self):
    self.input_embedding = tf.compat.v2.keras.layers.Embedding(input_dim=self.hparams.input_dim,
                                                               output_dim=self.hparams.embedding_dim,
                                                               input_shape=(None, None),
                                                               mask_zero=True,
                                                               embeddings_regularizer=self.regularizer,
                                                               name='input_embedding')
    self.input_embedding_dropout = tf.keras.layers.Dropout(self.hparams.input_dropout_rate)
    self.output_embedding_dropout = tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate)

    self.output_embedding = tf.compat.v2.keras.layers.Dense(units=self.hparams.output_dim,
                                                            kernel_regularizer=self.regularizer,
                                                            bias_regularizer=self.regularizer,
                                                            name='output_projection')

    self.stacked_rnns = []
    for _ in np.arange(self.hparams.depth):
      self.stacked_rnns.append(tf.keras.layers.LSTM(units=self.hparams.hidden_dim,
                                                    return_sequences=True,
                                                    return_state=True,
                                                    go_backwards=False,
                                                    stateful=False,
                                                    unroll=False,
                                                    time_major=False,
                                                    recurrent_dropout=self.hparams.hidden_dropout_rate,
                                                    dropout=self.hparams.hidden_dropout_rate,
                                                    kernel_regularizer=self.regularizer,
                                                    recurrent_regularizer=self.regularizer,
                                                    bias_regularizer=self.regularizer,

                                                    ))

  def call(self, inputs, **kwargs):
    if 'training' in kwargs:
      training = kwargs['training']
    else:
      training = False

    @tf.function(experimental_relax_shapes=True)
    def _call(inputs, training):
      embedded_input = self.input_embedding_dropout(self.input_embedding(inputs),training=training)
      rnn_outputs = embedded_input

      input_mask = self.input_embedding.compute_mask(inputs)
      inputs_length = tf.reduce_sum(tf.cast(input_mask, dtype=tf.int32), axis=-1)

      for i in np.arange(self.hparams.depth):
        rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs, mask=input_mask, training=training)

      rnn_outputs = self.output_embedding_dropout(rnn_outputs, training=training)
      batch_size = tf.shape(rnn_outputs)[0]
      bach_indices = tf.expand_dims(tf.range(batch_size), 1)
      final_indexes = tf.concat([bach_indices, tf.expand_dims(tf.cast(inputs_length - 1, dtype=tf.int32), 1)], axis=-1)


      final_rnn_outputs = tf.gather_nd(rnn_outputs, final_indexes)

      logits = self.output_embedding(final_rnn_outputs)
      return logits

    return _call(inputs, training)


class LmLSTMSharedEmb(tf.keras.Model):

  def __init__(self, hparams, scope="lm_lstm_shared_emb",*inputs, **kwargs):
    del kwargs['cl_token']
    super(LmLSTMSharedEmb, self).__init__()
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'em-'+str(self.hparams.embedding_dim),
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.0000)
    self.create_vars()

  def create_vars(self):

    @tf.function
    def _create_vars():
      self.input_embedding = SharedEmbeddings(vocab_size=self.hparams.input_dim,
                                              hidden_size=self.hparams.embedding_dim,
                                              initializer_range=self.hparams.initializer_range,
                                              regularizer=self.regularizer,
                                              name='embedding')
      self.input_embedding_dropout = tf.keras.layers.Dropout(self.hparams.input_dropout_rate)
      self.output_embedding_dropout = tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate)
      initializer_range = self.hparams.embedding_dim ** -0.5 if self.hparams.initializer_range is None else self.hparams.initializer_range
      self.output_projection = tf.keras.layers.Dense(units=self.hparams.embedding_dim,
                                                     kernel_initializer=get_initializer(initializer_range))

      self.stacked_rnns = []
      self.rnn_initial_states = []
      for _ in np.arange(self.hparams.depth):
        initializer_range = self.hparams.hidden_dim ** -0.5 if self.hparams.initializer_range is None else self.hparams.initializer_range
        self.stacked_rnns.append(tf.keras.layers.LSTM(units=self.hparams.hidden_dim,
                                                      return_sequences=True,
                                                      return_state=True,
                                                      go_backwards=False,
                                                      stateful=False,
                                                      unroll=False,
                                                      time_major=False,
                                                      recurrent_dropout=self.hparams.hidden_dropout_rate,
                                                      dropout=self.hparams.hidden_dropout_rate,
                                                      kernel_regularizer=self.regularizer,
                                                      recurrent_regularizer=self.regularizer,
                                                      bias_regularizer=self.regularizer,
                                                      kernel_initializer=get_initializer(initializer_range),
                                                      recurrent_initializer=get_initializer(initializer_range)
                                                      ))

    _create_vars()
    initializer_range = self.hparams.hidden_dim ** -0.5 if self.hparams.initializer_range is None else self.hparams.initializer_range
    for i in np.arange(self.hparams.depth):
      state_size = self.stacked_rnns[i].cell.state_size
      if nest.is_sequence(state_size):
        init_state = nest.map_structure(lambda x: create_init_var(x, i, initializer_range), state_size)
      else:
        init_state = create_init_var(state_size, i, initializer_range)

      self.rnn_initial_states.append(init_state)



  def call(self, inputs, padding_symbol=0, **kwargs):
    @tf.function(experimental_relax_shapes=True)
    def _call(inputs, padding_symbol, **kwargs):
      input_mask = tf.cast(inputs != padding_symbol, dtype=tf.bool)
      embedded_input = self.input_embedding_dropout(self.input_embedding(inputs, mode='embedding'),
                                                    **kwargs)
      rnn_outputs = embedded_input
      for i in np.arange(self.hparams.depth):
        batch_size_tensor = tf.shape(rnn_outputs)[0]
        absl.logging.info(self.rnn_initial_states[i])

        def tile_init(unnested_init_state):
          return tf.tile(unnested_init_state, (batch_size_tensor, 1))

        init_state = self.rnn_initial_states[i]
        if nest.is_sequence(init_state):
          init_for_batch = nest.map_structure(tile_init, init_state)
        else:
          init_for_batch = tile_init(init_state)


        rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs, mask=input_mask,
                                                             initial_state=init_for_batch,
                                                             **kwargs)

      rnn_outputs = self.output_projection(rnn_outputs, **kwargs)
      rnn_outputs = self.output_embedding_dropout(rnn_outputs,**kwargs)
      logits = self.input_embedding(rnn_outputs, mode='linear')

      return logits

    return _call(inputs, padding_symbol, **kwargs)



if __name__ == '__main__':
  class hparams(object):
    hidden_dim=8
    input_dim=4
    output_dim=4
    depth=2
    hidden_dropout_rate=0.1

  lm_lstm = LmLSTM(hparams=hparams)
  inputs = np.int64(np.flip(np.sort(np.random.uniform(0,3,size=(2,5)))))
  inputs_mask = tf.equal(inputs, 0)
  print(inputs_mask)
  lm_lstm.build(input_shape=(None,None))
  lm_lstm.summary()

  print(inputs)
  print(lm_lstm(inputs))
