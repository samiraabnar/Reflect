import absl
import tensorflow as tf
import numpy as np

from tf2_models.common_layers import get_initializer
from tf2_models.embedding import SharedEmbeddings


class LmLSTM(tf.keras.Model):

  def __init__(self, hparams, scope="lm_lstm"):
    super(LmLSTM, self).__init__()
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'em-'+str(self.hparams.embedding_dim),
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00000)
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

  def __init__(self, hparams, scope="cl_lstm"):
    super(ClassifierLSTM, self).__init__()
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'em-'+str(self.hparams.embedding_dim),
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])

    self.regularizer = tf.keras.regularizers.l1_l2(l1=0.00,
                                                   l2=0.00000)
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


class LmLSTMSharedEmb(tf.keras.Model):

  def __init__(self, hparams, scope="lm_lstm_shared_emb"):
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
    self.input_embedding = SharedEmbeddings(vocab_size=self.hparams.input_dim ,
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

      print(self.hparams.hidden_dim)
      random_value = tf.random.normal(shape=(self.hparams.hidden_dim,), dtype=tf.float32)
      self.rnn_initial_states.append(tf.Variable(initial_value=random_value))


  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, padding_symbol=0, **kwargs):
    input_mask = tf.cast(inputs != padding_symbol, dtype=tf.bool)
    float_input_mask= tf.cast(input_mask, dtype=tf.float32)
    embedded_input = self.input_embedding_dropout(self.input_embedding(inputs, mode='embedding'),
                                                  **kwargs)
    rnn_outputs = embedded_input
    for i in np.arange(self.hparams.depth):
      absl.logging.info(self.rnn_initial_states[i])
      init_state = self.stacked_rnns[i].get_initial_state(rnn_outputs)
      print(init_state)
      #init_state = tf.multiply(init_state, self.rnn_initial_states[i])
      rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs, mask=input_mask,
                                                           initial_state=init_state,
                                                           **kwargs)

    rnn_outputs = self.output_projection(rnn_outputs, **kwargs)
    rnn_outputs = self.output_embedding_dropout(rnn_outputs,**kwargs)
    logits = self.input_embedding(rnn_outputs, mode='linear')
    logits = logits * float_input_mask[...,None] + tf.eye(self.hparams.output_dim)[0] * (1 - float_input_mask[...,None])

    return logits



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
