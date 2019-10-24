import tensorflow as tf



@tf.function(experimental_relax_shapes=True)
def get_masked_probs(logits, labels, temperature, padding_symbol=0):
  teacher_probs = tf.nn.softmax(logits / temperature, axis=-1)
  sequence_mask = tf.cast(labels != padding_symbol, dtype=tf.float32)
  masked_teacher_probs = teacher_probs * sequence_mask[..., None] + tf.eye(tf.shape(teacher_probs)[-1])[0] * (
      1 - sequence_mask[..., None])

  return masked_teacher_probs


@tf.function(experimental_relax_shapes=True)
def get_probs(logits, labels, temperature):
  teacher_probs = tf.nn.softmax(logits / temperature, axis=-1)

  return teacher_probs

class DistillLoss(tf.keras.losses.Loss):
  def __init__(self, padding_symbol=0, tmp=1.0,
               **kwargs):
    super(DistillLoss, self).__init__(**kwargs)
    self.tmp = tf.constant(tmp, dtype=tf.float32)
    self.padding_symbol = tf.constant(padding_symbol, dtype=tf.int32)

  def call(self, y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
    sequence_mask = tf.cast(y_true[..., self.padding_symbol] != 1.0, dtype=tf.float32)
    sequence_mask = sequence_mask / tf.reduce_sum(sequence_mask)
    return tf.reduce_sum(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_pred / self.tmp,
                                                                           labels=y_true,
                                                                           name='loss') * sequence_mask)
