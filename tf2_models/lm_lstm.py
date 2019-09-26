import tensorflow as tf
import numpy as np

class LmLSTM(tf.keras.Model):
  def __init__(self, hparams, scope="lm_lstm"):
    super(LmLSTM, self).__init__()
    self.hparams = hparams
    self.scope = scope

    self.model_name = '_'.join(self.scope,
                         'h-'+self.hparams.hidden_dim,
                         'd-'+self.hparams.depth,
                         'hdrop-'+self.hparams.hidden_dropout_rate,
                         'indrop-'+self.hparams.input_dropout_rate)



    self.input_embedding = tf.compat.v2.keras.layers.Embedding(input_dim=self.hparams.input_dim,
                                                               output_dim=self.hparams.hidden_dim,
                                                               input_shape=(None,None),
                                                               mask_zero=True,
                                                               name='input_embedding')
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

    self.build(input_shape=(None, None))
    self.summary()


  def call(self, inputs, **kwargs):
    embedded_input = self.input_embedding(inputs)
    rnn_outputs = embedded_input
    for i in np.arange(self.hparams.depth):
      rnn_outputs, state_h, state_c = self.stacked_rnns[i](rnn_outputs)

    logits = self.output_embedding(rnn_outputs)

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
