import tensorflow as tf

from tf2_models.metrics import distill_loss, sequence_distill_loss

@tf.function(experimental_relax_shapes=True)
def get_topk_mask(inputs, k):
  inputs_shape = tf.shape(inputs)
  inputs_shape = tf.cast(inputs_shape, dtype=tf.int64)

  values, indices = tf.nn.top_k(inputs, k=k, sorted=False)
  indices = tf.cast(indices, dtype=tf.int64)
  k = tf.cast(k, dtype=tf.int64)
  temp_indices = tf.meshgrid(*[tf.range(d, dtype=tf.int64) for d in (tf.unstack(
    inputs_shape[:(inputs.get_shape().ndims - 1)]) + [k])], indexing='ij')
  temp_indices = tf.stack(temp_indices[:-1] + [indices], axis=-1)
  full_indices = tf.reshape(temp_indices, [-1, inputs.get_shape().ndims])
  values = tf.reshape(values, [-1])

  mask_vals = tf.ones_like(values, dtype=tf.int64)


  full_indices = tf.cast(
    full_indices, dtype=tf.int64)
  mask_st = tf.SparseTensor(indices=full_indices, values=mask_vals, dense_shape=inputs_shape)
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
    self.padding_symbol = tf.Variable(padding_symbol, dtype=tf.int64, name="padding_symbol")

  def call(self, y_true, y_pred):
    return distill_loss(y_true, y_pred, self.tmp)


class SequenceDistillLoss(tf.keras.losses.Loss):
  def __init__(self, padding_symbol=0, tmp=1.0,
               **kwargs):
    super(SequenceDistillLoss, self).__init__(**kwargs)
    self.tmp = tf.Variable(tmp, dtype=tf.float32, name="tmp")
    self.padding_symbol = tf.Variable(padding_symbol, dtype=tf.int64, name="padding_symbol")

  def call(self, y_true, y_pred):
    return sequence_distill_loss(y_true, y_pred, self.padding_symbol, self.tmp)


def get_distill_scheduler(schedule, min=0.0, max=1.0, decay_steps=10000):
  if schedule is "exp":
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
      max,
      decay_steps=1000,
      decay_rate=0.96,
      staircase=True)
  elif schedule is 'crs':
    scheduler = tf.keras.experimental.CosineDecayRestarts(
      max,
      decay_steps,
      t_mul=2.0,
      m_mul=0.9,
      alpha=0.001,
    )
  elif schedule is 'lnr':
    a = (max - min) / decay_steps
    scheduler = lambda x: max - a*x
  elif schedule is 'stp':
    scheduler = lambda x: max if x < decay_steps else min
  else:
    scheduler = lambda x: max

  return scheduler