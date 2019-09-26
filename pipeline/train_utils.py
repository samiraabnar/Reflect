import tensorflow as tf

@tf.function
def train(model, dataset, optimizer, loss_fn, avg_metric_dic, task):
  for examples in dataset:
    feature_dic = task.convert_examples(examples)
    x, y = feature_dic['inputs'], feature_dic['targets']
    with tf.GradientTape() as tape:
      inputs_mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
      logits = model(x)
      loss = loss_fn(logits, y, inputs_mask)
      for metric in avg_metric_dic:
        avg_metric_dic[metric].update_state(task.metrics[metric](logits, y, inputs_mask))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  t_loss = avg_metric_dic['loss'].result()
  for metric in avg_metric_dic:
    tf.summary.scalar('loss', avg_metric_dic[metric].result(), step=optimizer.iterations)
    avg_metric_dic[metric].reset_states()

  return t_loss

@tf.function
def eval(model, dataset, loss_fn, avg_metric_dic, task, step_num):
  for examples in dataset:
    feature_dic = task.convert_examples(examples)
    x, y = feature_dic['inputs'], feature_dic['targets']
    inputs_mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
    logits = model(x)
    loss = loss_fn(logits, y, inputs_mask)
    for metric in avg_metric_dic:
      avg_metric_dic[metric].update_state(task.metrics[metric](logits, y, inputs_mask))

  for metric in avg_metric_dic:
    tf.summary.scalar('loss', avg_metric_dic[metric].result(), step=step_num)
    avg_metric_dic[metric].reset_states()