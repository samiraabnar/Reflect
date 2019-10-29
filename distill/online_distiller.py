import tensorflow as tf
import os

from distill.distiller import Distiller
from tf2_models.train_utils import ExponentialDecayWithWarmpUp
from tf2_models.trainer import OPTIMIZER_DIC
from tf2_models.utils import log_summary, camel2snake
from inspect import isfunction

class OnlineDistiller(Distiller):
  def __init__(self, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.distill_params = distill_params
    self.temperature = tf.convert_to_tensor(distill_params.distill_temp)

    self.create_student_optimizer()
    self.create_teacher_optimizer()

    self.setup_ckp_and_summary(student_ckpt_dir, student_log_dir, teacher_ckpt_dir, teacher_log_dir)
    self.setup_models(distill_params, task)
    self.setup_loggings()

  def create_teacher_optimizer(self):
    teacher_initial_learning_rate = self.distill_params.teacher_learning_rate
    lr_schedule = ExponentialDecayWithWarmpUp(
      initial_learning_rate=teacher_initial_learning_rate,
      decay_steps=self.distill_params.teacher_decay_steps,
      decay_rate=0.96,
      warmup_steps=self.distill_params.teacher_warmup_steps)
    self.teacher_optimizer = OPTIMIZER_DIC[self.distill_params.teacher_optimizer](
      learning_rate=lr_schedule, epsilon=1e-08, clipnorm=1.0)

  def setup_loggings(self):
    self.student_validation_metrics = {}
    for metric in self.task.metrics():
      if isfunction(metric):
        self.student_validation_metrics[camel2snake(metric.__name__)] = tf.keras.metrics.Mean()
      else:
        self.student_validation_metrics[camel2snake(metric.__class__.__name__)] = tf.keras.metrics.Mean()
    self.student_validation_loss = tf.keras.metrics.Mean()

    self.teacher_validation_metrics = {}
    for metric in self.task.metrics():
      if isfunction(metric):
        self.teacher_validation_metrics[camel2snake(metric.__name__)] = tf.keras.metrics.Mean()
      else:
        self.teacher_validation_metrics[camel2snake(metric.__class__.__name__)] = tf.keras.metrics.Mean()

  def setup_models(self, distill_params, task):
    x, y = iter(self.task.valid_dataset).next()
    self.student_model(x)
    self.student_model.summary()
    self.teacher_model(x)
    self.teacher_model.summary()

    self.student_model.compile(
      optimizer=self.student_optimizer,
      loss=task.get_distill_loss_fn(distill_params),
      metrics=[task.metrics()])

    self.teacher_model.compile(
      optimizer=self.teacher_optimizer,
      loss=task.get_loss_fn(),
      metrics=[task.metrics()])

  @tf.function(experimental_relax_shapes=True)
  def teacher_train_step(self, x, y_true):
    with tf.GradientTape() as tape:
      logits = self.teacher_model(x)
      loss = self.teacher_model.loss(y_pred=logits, y_true=y_true)

    grads = tape.gradient(loss, self.teacher_model.trainable_weights)
    self.teacher_model.optimizer.apply_gradients(zip(grads, self.teacher_model.trainable_weights))

    return logits, loss

  def distill_loop(self, padding_symbol=0):
    with self.summary_writer.as_default():
      train_iter = iter(self.task.train_dataset)
      valid_iter = iter(self.task.valid_dataset)

      step = 0
      epochs = 0
      num_epochs = self.distill_params.n_epochs
      for (x, y) in train_iter:
        x = tf.convert_to_tensor(x, dtype=tf.int64)
        y = tf.convert_to_tensor(y, dtype=tf.int64)

        teacher_logits, teacher_loss = self.teacher_train_step(x, y)
        teacher_probs = self.task.get_probs_fn()(logits=teacher_logits, labels=y, temperature=self.temperature)
        distill_loss, actual_loss = self.student_train_step(x=x, y=teacher_probs, y_true=y)

        # Log every 200 batches.
        if step % 200 == 0:
          log_summary(log_name='student_learning_rate',
                      log_value=self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations),
                      summary_scope='train')
          log_summary(log_name='teacher_learning_rate',
                      log_value=self.teacher_model.optimizer.learning_rate(self.teacher_model.optimizer.iterations),
                      summary_scope='train')
          log_summary(log_name='fine_distill_loss', log_value=distill_loss, summary_scope='train')
          log_summary(log_name='teacher_loss', log_value=teacher_loss, summary_scope='train')

        # Checkpoint and log after each epoch
        if (step % self.task.n_train_batches) == 0:
          tf.print("Epoch %d, distill loss:" %epochs, distill_loss)
          self.validate(actual_loss, distill_loss, valid_iter)
          self.save_student()
          self.save_teacher()
          epochs += 1

        step += 1
        # Stop, if reached the number of training epochs
        if epochs >= num_epochs:
          break

  def save_teacher(self):
    self.teacher_ckpt.step.assign_add(1)
    save_path = self.teacher_manager.save()
    tf.print("Saved teacher checkpoint for step {}: {}".format(int(self.teacher_ckpt.step), save_path))


  def validate(self, actual_loss, distill_loss, valid_iter):
    ''' Offline Distillation main loop.
    '''

    tf.print('Validating ...')
    valid_step = 0
    for v_x, v_y in valid_iter:
      v_x = tf.convert_to_tensor(v_x, dtype=tf.int64)
      v_y = tf.convert_to_tensor(v_y, dtype=tf.int64)

      teacher_logits = self.teacher_model(v_x)
      teacher_probs = self.task.get_probs_fn()(logits=teacher_logits, labels=v_y, temperature=self.temperature)
      logits = self.student_model(v_x)

      valid_step += 1
      for metric in self.task.metrics():
        if isfunction(metric):
          metric_name = camel2snake(metric.__name__)
        else:
          metric_name = camel2snake(metric.__class__.__name__)
        self.student_validation_metrics[metric_name].update_state(metric(y_pred=logits,
                                                                 y_true=v_y))
        self.teacher_validation_metrics[metric_name].update_state(metric(y_pred=teacher_logits,
                                                                         y_true=v_y))
        self.student_validation_loss.update_state(
          self.task.get_distill_loss_fn(self.distill_params)(y_true=teacher_probs, y_pred=logits))

      if valid_step >= self.task.n_valid_batches:
        break

    log_summary(log_name='distill_loss', log_value=distill_loss, summary_scope='train')
    log_summary(log_name='actual_loss', log_value=actual_loss, summary_scope='train')

    for metric in self.task.metrics():
      if isfunction(metric):
        metric_name = camel2snake(metric.__name__)
      else:
        metric_name = camel2snake(metric.__class__.__name__)
      log_summary(log_name=metric_name, log_value=self.student_validation_metrics[metric_name].result(), summary_scope='student_valid')
      log_summary(log_name=metric_name, log_value=self.teacher_validation_metrics[metric_name].result(), summary_scope='student_valid')

      self.student_validation_metrics[metric_name].reset_states()
      self.teacher_validation_metrics[metric_name].reset_states()


    log_summary(log_name="distill_loss", log_value=self.student_validation_loss.result(), summary_scope='student_valid')
    self.student_validation_loss.reset_states()

