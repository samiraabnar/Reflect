import tensorflow as tf
import re

from tensorboard.compat.tensorflow_stub import tensor_shape


def camel2snake(name):
  return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: '_' + x.group(0).lower(), name[1:])

def log_summary(log_value, log_name, summary_scope):
  """Produce scalar summaries."""
  with tf.compat.v2.summary.experimental.summary_scope(summary_scope):
    tf.summary.scalar(log_name, log_value)


def create_init_var(unnested_state_size, i, initializer_range):
  flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
  init_state_size = [1] + flat_dims
  return tf.Variable(shape=init_state_size, dtype=tf.float32,
                     initial_value=tf.keras.initializers.TruncatedNormal(stddev=initializer_range)(
                       shape=init_state_size),
                     trainable=True,
                     name="lstm_init_" + str(i))



