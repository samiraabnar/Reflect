import tensorflow as tf

from tf2_models.metrics import distill_loss, sequence_distill_loss

@tf.function(experimental_relax_shapes=True)
def get_topk_mask(inputs, k):
  values, indices = tf.nn.top_k(inputs, k=k, sorted=False)

  temp_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(
    tf.shape(inputs)[:(inputs.get_shape().ndims - 1)]) + [k])], indexing='ij')
  temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
  full_indices = tf.reshape(temp_indices, [-1, inputs.get_shape().ndims])
  values = tf.reshape(values, [-1])

  mask_st = tf.SparseTensor(indices=tf.cast(
    full_indices, dtype=tf.int64), values=tf.ones_like(values), dense_shape=inputs.shape)
  mask = tf.sparse.to_dense(tf.sparse.reorder(mask_st))

  return mask

@tf.function(experimental_relax_shapes=True)
def get_topk_masked_probs(logits, labels, temperature, k=100, padding_symbol=0):
  topk_mask = (1 - tf.cast(get_topk_mask(logits, k), dtype=tf.float32)) * -10e8
  teacher_probs = tf.nn.softmax((logits + topk_mask) / temperature, axis=-1)
  sequence_mask = tf.cast(labels != padding_symbol, dtype=tf.float32)
  masked_teacher_probs = teacher_probs * sequence_mask[..., None] + tf.eye(tf.shape(teacher_probs)[-1])[0] * (
      1 - sequence_mask[..., None])

  return masked_teacher_probs

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
    self.tmp = tf.Variable(tmp, dtype=tf.float32, name="temp")
    self.padding_symbol = tf.Variable(padding_symbol, dtype=tf.int32, name="padding_symbol")

  def call(self, y_true, y_pred):
    return distill_loss(y_true, y_pred, self.tmp)


class SequenceDistillLoss(tf.keras.losses.Loss):
  def __init__(self, padding_symbol=0, tmp=1.0,
               **kwargs):
    super(SequenceDistillLoss, self).__init__(**kwargs)
    self.tmp = tf.Variable(tmp, dtype=tf.float32, name="tmp")
    self.padding_symbol = tf.Variable(padding_symbol, dtype=tf.int32, name="padding_symbol")

  def call(self, y_true, y_pred):
    return sequence_distill_loss(y_true, y_pred, self.padding_symbol, self.tmp)