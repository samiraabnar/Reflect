import tensorflow as tf

@tf.function
def masked_sequence_loss(logits, targets, sequence_mask):
  return tf.reduce_mean(tf.compat.v2.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=targets,
                                                                  name='loss') * sequence_mask)
@tf.function
def accuracy(logits, targets, sequence_mask):
  return accuracy_topk(logits, targets, sequence_mask, topk=1)

@tf.function
def accuracy_top2(logits, targets, sequence_mask):
  return accuracy_topk(logits, targets, sequence_mask, topk=2)

@tf.function
def accuracy_top5(logits, targets, sequence_mask):
  return accuracy_topk(logits, targets, sequence_mask, topk=5)

@tf.function
def accuracy_topk(logits, targets, sequence_mask, topk):
  orig_shape = tf.shape(logits)
  last_dim = orig_shape[-1]
  logits = tf.reshape(logits, (-1,last_dim))
  targets = tf.reshape(targets, (-1,1))
  sequence_mask = tf.cast(tf.reshape(sequence_mask, (1,-1)), tf.float32)
  unmasked_accuracies = tf.metrics.top_k_categorical_accuracy(y_true=targets,
                                               y_pred=logits,
                                               k=topk)

  return tf.reduce_mean(sequence_mask * unmasked_accuracies)

if __name__ == '__main__':
  import numpy as np
  a = np.asarray([[[1,1.5,2,0], [4,3,0,0]],
                  [[1,1.5,2,0], [4,3,0,0]]], dtype=np.float32)
  a_mask = [[1, 1],[1 , 0]]
  print(a_mask)
  b = np.asarray([[0, 0],[1, 1]], dtype=np.int32)

  print(accuracy_topk(a,b,a_mask,3))