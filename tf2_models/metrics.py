import tensorflow as tf


@tf.function
def distill_loss(y_true, y_pred, padding_symbol, tmp):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
  sequence_mask = tf.cast(y_true[..., padding_symbol] != 1.0, dtype=tf.float32)
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
  return tf.reduce_sum(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_pred / tmp,
                                                                         labels=y_true,
                                                                         name='loss') * sequence_mask)

@tf.function
def masked_sequence_loss_with_probs(y_true, y_pred, padding_symbol=0):
  #y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
  sequence_mask = tf.cast(y_true[...,padding_symbol] != 1.0, dtype=tf.float32)
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask, axis=-1)
  return tf.reduce_mean(tf.reduce_sum(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_true,
                                                                  labels=y_pred,
                                                                  name='loss') * sequence_mask, axis=-1))

@tf.function
def sequence_loss_with_probs(y_true, y_pred):
  #y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
  return tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_true,
                                                                  labels=y_pred,
                                                                  name='loss'))

@tf.function
def masked_sequence_loss(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask, axis=-1)[...,None]
  return tf.reduce_mean(tf.reduce_sum(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss') * sequence_mask, axis=-1))

@tf.function
def batch_masked_sequence_loss(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
  return tf.reduce_sum(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss') * sequence_mask)

@tf.function
def masked_perplexity(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask, axis=-1)[...,None]
  return tf.reduce_mean(tf.exp(tf.reduce_sum(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss') * sequence_mask, axis=-1)))

@tf.function
def masked_batch_perplexity(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
  return tf.exp(tf.reduce_sum(sequence_mask * tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss')))

@tf.function
def sequence_loss(y_true, y_pred):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
  return tf.reduce_mean(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss'))


@tf.function
def accuracy(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=1)

@tf.function
def unmasked_accuracy(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  return unmasked_accuracy_topk(targets, logits, topk=1)

@tf.function
def accuracy_top2(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=2)

@tf.function
def unmasked_accuracy_top2(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  return unmasked_accuracy_topk(targets, logits, topk=2)

@tf.function
def accuracy_top5(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=5)

@tf.function
def unmasked_accuracy_top5(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int32)
  return unmasked_accuracy_topk(targets, logits, topk=5)

@tf.function
def accuracy_topk(targets, logits, sequence_mask, topk):
  orig_shape = tf.shape(logits)
  last_dim = orig_shape[-1]
  logits = tf.reshape(logits, (-1,last_dim))
  targets = tf.reshape(targets, (-1,1))
  sequence_mask = tf.cast(tf.reshape(sequence_mask, (-1,1)), tf.float32)
  unmasked_accuracies = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=targets,
                                               y_pred=logits,
                                               k=topk)
  return tf.reduce_mean(sequence_mask * unmasked_accuracies)

@tf.function
def unmasked_accuracy_topk(targets, logits, topk):
  orig_shape = tf.shape(logits)
  last_dim = orig_shape[-1]
  logits = tf.reshape(logits, (-1,last_dim))
  targets = tf.reshape(targets, (-1,1))
  unmasked_accuracies = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=targets,
                                               y_pred=logits,
                                               k=topk)
  return tf.reduce_mean(unmasked_accuracies)


class MaskedSequenceLoss(tf.keras.losses.Loss):
  def __init__(self, padding_symbol=0,
               **kwargs):
    super(MaskedSequenceLoss, self).__init__(**kwargs)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int32)
    self.name = "batch_masked_sequence_loss"

  def call(self, y_true, y_pred):
    return batch_masked_sequence_loss(y_true, y_pred, self.padding_symbol)

if __name__ == '__main__':
  import numpy as np
  a = np.asarray([[[1,1.5,2,0], [4,3,0,0]],
                  [[1,1.5,2,0], [4,3,0,0]]], dtype=np.float32)
  a_mask = [[1, 1],[1 , 0]]
  print(a_mask)
  b = np.asarray([[0, 0],[1, 1]], dtype=np.int32)

  print(accuracy_topk(logits=a,targets=b,sequence_mask=a_mask,topk=1))