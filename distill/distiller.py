import tensorflow as tf

from distill.distill_util import get_probs
from tf2_models.metrics import masked_sequence_loss
from tf2_models.utils import log_summary


class Distiller(object):
  def __init__(self, distill_params, teacher_model, student_model, task):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.hparams = distill_params
    self.temperature = tf.convert_to_tensor(distill_params.distill_temp)



  def distill_loop(self, padding_symbol=0):
    @tf.function(experimental_relax_shapes=True)
    def get_logits(x):
      return self.student_model(x)

    @tf.function(experimental_relax_shapes=True)
    def train_step(x, y, y_true):
      with tf.GradientTape() as tape:
        logits = self.student_model(x)
        distill_loss = self.student_model.loss(y_pred=logits, y_true=y)

      grads = tape.gradient(distill_loss, self.student_model.trainable_weights)
      self.student_model.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))
      actual_loss = masked_sequence_loss(y_pred=logits, y_true=y_true)
      return distill_loss, actual_loss

    train_iter = iter(self.task.train_dataset)
    valid_iter = iter(self.task.valid_dataset)

    step = 0
    epochs = 0
    num_epochs = 3
    for (x, y) in train_iter:
      x = tf.convert_to_tensor(x, dtype=tf.int64)
      y = tf.convert_to_tensor(y, dtype=tf.int64)

      teacher_logits = self.teacher_model(x)
      masked_teacher_probs = get_probs(logits=teacher_logits, labels=y, temperature=self.temperature)

      distill_loss, actual_loss = train_step(x, masked_teacher_probs, y)
      # Log every 200 batches.
      if step % 200 == 0:
        log_summary(log_name='learning_rate',
                    log_value=self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations),
                    summary_scope='train')
        log_summary(log_name='fine_distill_loss', log_value=distill_loss, summary_scope='train')

      if (step % self.task.n_train_batches) == 0:
        valid_step = 0
        validation_actual_loss = tf.keras.metrics.Mean()
        validation_distill_loss = tf.keras.metrics.Mean()
        while valid_step < self.task.n_valid_baches:
          v_x, v_y = next(valid_iter)
          v_x = tf.convert_to_tensor(v_x, dtype=tf.int64)
          v_y = tf.convert_to_tensor(v_y, dtype=tf.int64)

          teacher_logits = self.teacher_model(v_x)
          masked_teacher_probs = get_probs(logits=teacher_logits,
                                           labels=v_y,
                                           temperature=self.temperature)

          logits = self.student_model(x)
          validation_distill_loss.update_state(self.student_model.loss(y_pred=logits,
                                                                       y_true=masked_teacher_probs))
          validation_actual_loss.update_state(masked_sequence_loss(y_true=v_y,
                                                                   y_pred=logits))
          valid_step += 1

        log_summary(log_name='distill_loss', log_value=distill_loss, summary_scope='train')
        log_summary(log_name='actual_loss', log_value=actual_loss, summary_scope='train')
        log_summary(log_name='perolexity', log_value=tf.exp(actual_loss), summary_scope='train')
        log_summary(log_name='distill_loss', log_value=validation_distill_loss.result(), summary_scope='valid')
        log_summary(log_name='actual_loss', log_value=validation_actual_loss.result(), summary_scope='valid')
        log_summary(log_name='perplexity', log_value=tf.exp(validation_actual_loss.result()), summary_scope='valid')

        epochs += 1

      step += 1

      if epochs >= num_epochs:
        break

  def run(self):
    raise NotImplementedError
