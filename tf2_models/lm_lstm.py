import tensorflow as tf
import numpy as np

class LmLSTM(tf.keras.Model):

  def __init__(self, hparams, scope="lm_lstm"):
    super(LmLSTM, self).__init__()
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join([self.scope,
                         'h-'+str(self.hparams.hidden_dim),
                         'd-'+str(self.hparams.depth),
                         'hdrop-'+str(self.hparams.hidden_dropout_rate),
                         'indrop-'+str(self.hparams.input_dropout_rate)])



    self.create_vars()

  @tf.function
  def create_vars(self):
    self.input_embedding = tf.compat.v2.keras.layers.Embedding(input_dim=self.hparams.input_dim,
                                                               output_dim=self.hparams.hidden_dim,
                                                               input_shape=(None, None),
                                                               mask_zero=True,
                                                               name='input_embedding')
    self.input_embedding_dropout = tf.keras.layers.Dropout(self.hparams.input_dropout_rate)
    self.output_embedding_dropout = tf.keras.layers.Dropout(self.hparams.hidden_dropout_rate)

    self.output_embedding = tf.compat.v2.keras.layers.Dense(units=self.hparams.output_dim,
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
                                                    dropout=self.hparams.hidden_dropout_rate
                                                    ))
  @tf.function
  def call(self, inputs, **kwargs):
    embedded_input = self.input_embedding_dropout(self.input_embedding(inputs))
    rnn_outputs = embedded_input


    input_mask = tf.cast(self.input_embedding.compute_mask(inputs), dtype=tf.float32)
    for i in np.arange(self.hparams.depth):
      rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs, mask=input_mask)

    rnn_outputs = self.output_embedding_dropout(rnn_outputs)
    logits = self.output_embedding(rnn_outputs)
    logits = logits * input_mask[...,None] + tf.eye(self.hparams.output_dim)[0] * (1 - input_mask[...,None])

    return logits

  def update(self, loss):
    pass


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
