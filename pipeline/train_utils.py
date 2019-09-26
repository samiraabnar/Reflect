import tensorflow as tf
import numpy as np

@tf.function
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
  """Cosine decay schedule with warm up period.
  Cosine annealing learning rate as described in:
    Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
    ICLR 2017. https://arxiv.org/abs/1608.03983
  In this schedule, the learning rate grows linearly from warmup_learning_rate
  to learning_rate_base for warmup_steps, then transitions to a cosine decay
  schedule.
  Args:
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
  Returns:
    If executing eagerly:
      returns a no-arg callable that outputs the (scalar)
      float tensor learning rate given the current value of global_step.
    If in a graph:
      immediately returns a (scalar) float tensor representing learning rate.
  Raises:
    ValueError: if warmup_learning_rate is larger than learning_rate_base,
      or if warmup_steps is larger than total_steps.
  """
  if total_steps < warmup_steps:
    raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
  def eager_decay_rate():
    """Callable to compute the learning rate."""
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
      learning_rate = tf.where(
          global_step > warmup_steps + hold_base_rate_steps,
          learning_rate, learning_rate_base)
    if warmup_steps > 0:
      if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger or equal to '
                         'warmup_learning_rate.')
      slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
      warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
      learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                               learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name='learning_rate')

  if tf.executing_eagerly():
    return eager_decay_rate
  else:
    return eager_decay_rate()

@tf.function
def train(model, dataset, optimizer, loss_fn, avg_metric_dic, task, inter_log_steps=1000):
  for examples in dataset:
    feature_dic = task.convert_examples(examples)
    x, y = feature_dic['inputs'], feature_dic['targets']
    with tf.GradientTape() as tape:
      inputs_mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
      logits = model(x)
      loss = loss_fn(logits, y, inputs_mask)
      if (optimizer.iterations % inter_log_steps) == 0:
        tf.print("eager loss is: ", loss)
        #tf.summary.scalar('eager_loss', loss, step=optimizer.iterations)
        #tf.summary.scalar('learning_rate', optimizer.learning_rate, step=optimizer.iterations)
      for metric in avg_metric_dic:
        avg_metric_dic[metric].update_state(task.metrics[metric](logits, y, inputs_mask))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  t_loss = avg_metric_dic['loss'].result()
  return t_loss

@tf.function
def eval(model, dataset, avg_metric_dic, task, step_num):
  for examples in dataset:
    feature_dic = task.convert_examples(examples)
    x, y = feature_dic['inputs'], feature_dic['targets']
    inputs_mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
    logits = model(x)
    for metric in avg_metric_dic:
      avg_metric_dic[metric].update_state(task.metrics[metric](logits, y, inputs_mask))
