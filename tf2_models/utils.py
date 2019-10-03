import tensorflow as tf

def log_summary(log_value, log_name, summary_scope):
  """Produce scalar summaries."""
  with tf.compat.v2.summary.experimental.summary_scope(summary_scope):
    tf.compat.v2.summary.scalar(log_name, log_value)


