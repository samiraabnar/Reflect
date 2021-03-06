import tensorflow as tf


@tf.function(experimental_relax_shapes=True)
def distill_loss(y_true, y_pred, tmp):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
  scale_factor = 1.0 / (tmp*tmp)
  return tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_pred / tmp,
                                                                         labels=y_true,
                                                                         name='loss')) * scale_factor

@tf.function(experimental_relax_shapes=True)
def sequence_distill_loss(y_true, y_pred, padding_symbol, tmp):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
  sequence_mask = tf.cast(y_true[..., padding_symbol] != 1.0, dtype=tf.float32)
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
  scale_factor =  1.0 / (tmp * tmp)
  return tf.reduce_sum(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_pred / tmp,
                                                                         labels=y_true,
                                                                         name='loss') * sequence_mask) * scale_factor


@tf.function(experimental_relax_shapes=True)
def masked_sequence_loss(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask, axis=-1)[...,None]
  return tf.reduce_mean(tf.reduce_sum(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss') * sequence_mask, axis=-1))

@tf.function(experimental_relax_shapes=True)
def batch_masked_sequence_loss(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask
  return tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss'), sequence_mask

@tf.function(experimental_relax_shapes=True)
def masked_perplexity(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask, axis=-1)[...,None]
  return tf.reduce_mean(tf.exp(tf.reduce_sum(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss') * sequence_mask, axis=-1)))

@tf.function(experimental_relax_shapes=True)
def masked_batch_perplexity(y_true, y_pred, padding_symbol=0):
  y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
  sequence_mask = tf.cast(y_true != padding_symbol, dtype=tf.float32)
  # [batch_size, 1]
  sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
  return tf.exp(tf.reduce_sum(sequence_mask * tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss')))

#@tf.function(experimental_relax_shapes=True)
def classification_loss(y_true, y_pred):
  if len(y_true.shape) > 1:
    y_true = tf.squeeze(y_true, axis=-1)

  y_true = tf.cast(y_true, dtype=tf.int64)

  return tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                  labels=y_true,
                                                                  name='loss')


@tf.function(experimental_relax_shapes=True)
def accuracy(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=tf.constant(1))

@tf.function(experimental_relax_shapes=True)
def unmasked_accuracy(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  return unmasked_accuracy_topk(targets, logits, topk=tf.constant(1))

@tf.function(experimental_relax_shapes=True)
def accuracy_top2(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=tf.constant(2))

@tf.function(experimental_relax_shapes=True)
def unmasked_accuracy_top2(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  return unmasked_accuracy_topk(targets, logits, topk=tf.constant(2))

@tf.function(experimental_relax_shapes=True)
def accuracy_top5(targets, logits, padding_symbol=0):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  sequence_mask = tf.cast(targets != padding_symbol, dtype=tf.float32)
  return accuracy_topk(targets, logits, sequence_mask, topk=tf.constant(5))

@tf.function(experimental_relax_shapes=True)
def unmasked_accuracy_top5(targets, logits, ):
  targets = tf.cast(tf.squeeze(targets), dtype=tf.int64)
  return unmasked_accuracy_topk(targets, logits, topk=tf.constant(5))

@tf.function(experimental_relax_shapes=True)
def accuracy_topk(targets, logits, sequence_mask, topk):
  orig_shape = tf.shape(logits)
  last_dim = orig_shape[-1]
  logits = tf.reshape(logits, (-1,last_dim))
  targets = tf.reshape(targets, (-1,1))
  sequence_mask = tf.cast(tf.reshape(sequence_mask, (-1,1)), tf.float32)
  unmasked_accuracies = tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true=targets,
                                               y_pred=logits,
                                               k=topk)
  normalizing_factor = sequence_mask / tf.reduce_sum(sequence_mask)
  normalizing_factor = tf.squeeze(normalizing_factor)

  return tf.reduce_sum(tf.multiply(normalizing_factor, unmasked_accuracies))

@tf.function(experimental_relax_shapes=True)
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
  def __init__(self, padding_symbol=0, num_replicas_in_sync=1,
               **kwargs):
    super(MaskedSequenceLoss, self).__init__(reduction=tf.keras.losses.Reduction.SUM, **kwargs)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int64)
    self.name = "batch_masked_sequence_loss"
    self.num_replicas_in_sync = num_replicas_in_sync

  def call(self, y_true, y_pred, sample_weight=None):
    entropies, mask = batch_masked_sequence_loss(y_true=y_true, y_pred=y_pred, padding_symbol=self.padding_symbol)
    if sample_weight is not None:
      mask = sample_weight

    norm_factor = mask / tf.reduce_sum(mask)
    return tf.reduce_sum(entropies * norm_factor) / self.num_replicas_in_sync


class MaskedSequenceMetric(tf.keras.losses.Loss):
  def __init__(self, padding_symbol=0,
               **kwargs):
    super(MaskedSequenceMetric, self).__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int64)
    self.name = "batch_masked_sequence_loss"

  def call(self, y_true, y_pred, sample_weight=None):
    entropies, mask = batch_masked_sequence_loss(y_true=y_true, y_pred=y_pred, padding_symbol=self.padding_symbol)
    if sample_weight is not None:
      mask = sample_weight

    norm_factor = mask / tf.reduce_sum(mask)
    return tf.reduce_sum(entropies * norm_factor)

class ClassificationLoss(tf.keras.losses.Loss):
  def __init__(self, global_batch_size, padding_symbol=tf.constant(0),
               **kwargs):
    super(ClassificationLoss, self).__init__(reduction=tf.keras.losses.Reduction.SUM, **kwargs)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int64)
    self.name = "classification_loss"
    self.global_batch_size = tf.cast(global_batch_size, dtype=tf.float32)


  def call(self, y_true, y_pred):
    return classification_loss(y_true=y_true, y_pred=y_pred) / self.global_batch_size

class ClassificationLossMetric(tf.keras.losses.Loss):
  def __init__(self, global_batch_size, padding_symbol=0,
               **kwargs):
    super(ClassificationLossMetric, self).__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int64)
    self.name = "classification_loss"
    self.global_batch_size = global_batch_size


  def call(self, y_true, y_pred):
    return tf.reduce_mean(classification_loss(y_true=y_true, y_pred=y_pred), axis=0)

class AccuracyTopk(tf.keras.losses.Loss):
  def __init__(self, global_batch_size, padding_symbol=0, topk=1,
               **kwargs):
    super(AccuracyTopk, self).__init__(reduction=tf.keras.losses.Reduction.NONE, **kwargs)
    self.name = '-'.join(['accuracy','top', str(topk)])
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int64)
    self.global_batch_size = global_batch_size
    self.topk = tf.constant(topk)


  def call(self, y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int64)
    sequence_mask = tf.cast(y_true != self.padding_symbol, dtype=tf.float32)
    return accuracy_topk(targets=y_true, logits=y_pred, sequence_mask=sequence_mask, topk=self.topk)


if __name__ == '__main__':
  import numpy as np
  a = np.asarray([[[1,1.5,2,0], [4,3,0,0]],
                  [[1,1.5,2,0], [4,3,0,0]]], dtype=np.float32)
  a_mask = [[1, 1],[1 , 0]]
  print(a_mask)
  b = np.asarray([[0, 0],[1, 1]], dtype=np.int64)

  print(accuracy_topk(logits=a,targets=b,sequence_mask=a_mask,topk=1))