import tensorflow as tf
import re

def camel2snake(name):
  return name[0].lower() + re.sub(r'(?!^)[A-Z]', lambda x: '_' + x.group(0).lower(), name[1:])

@tf.function(experimental_relax_shapes=True)
def log_summary(log_value, log_name, summary_scope):
  """Produce scalar summaries."""
  with tf.compat.v2.summary.experimental.summary_scope(summary_scope):
    tf.compat.v2.summary.scalar(log_name, log_value)


