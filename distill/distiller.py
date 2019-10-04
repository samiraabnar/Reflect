import tensorflow as tf
import os
from distill.distill_util import get_probs, DistillLoss
from tf2_models.metrics import masked_sequence_loss, masked_sequence_loss_with_probs
from tf2_models.train_utils import ExponentialDecayWithWarmpUp
from tf2_models.trainer import OPTIMIZER_DIC
from tf2_models.utils import log_summary


class Distiller(object):
  def __init__(self, distill_params, teacher_model, student_model, task,
               teacher_log_dir, student_log_dir, teacher_ckpt_dir, student_ckpt_dir):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.distill_params = distill_params
    self.temperature = tf.convert_to_tensor(distill_params.distill_temp)

    student_initial_learning_rate = self.distill_params.student_learning_rate
    lr_schedule = ExponentialDecayWithWarmpUp(
      initial_learning_rate=student_initial_learning_rate,
      decay_steps=self.distill_params.student_decay_steps,
      decay_rate=0.96,
      warmup_steps=self.distill_params.student_warmup_steps)

    self.student_optimizer = OPTIMIZER_DIC[self.distill_params.student_optimizer](
      learning_rate=lr_schedule, epsilon=1e-08, clipnorm=1.0)

    self.teacher_ckpt = tf.train.Checkpoint( net=self.teacher_model)
    self.teacher_manager = tf.train.CheckpointManager(self.teacher_ckpt, teacher_ckpt_dir, max_to_keep=2)

    self.student_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                            optimizer=self.student_optimizer,
                                            net=self.student_model)
    self.student_manager = tf.train.CheckpointManager(self.student_ckpt, teacher_ckpt_dir, max_to_keep=2)


    x, y = iter(self.task.valid_dataset).next()
    self.student_model(x)
    self.student_model.summary()

    self.teacher_model(x)
    self.teacher_model.summary()

    self.student_model.compile(
      optimizer=self.student_optimizer,
      loss=DistillLoss(tmp=distill_params.distill_temp),
      metrics=[DistillLoss(tmp=distill_params.distill_temp)])

    student_summary_dir = os.path.join(student_log_dir, 'summaries')
    tf.io.gfile.makedirs(student_log_dir)
    self.summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(student_summary_dir, 'train'))
    tf.compat.v2.summary.experimental.set_step(self.student_optimizer.iterations)


  def restore_teacher(self):
    self.teacher_ckpt.restore(self.teacher_manager.latest_checkpoint)
    if self.teacher_manager.latest_checkpoint:
      print("Restored teacher from {}".format(self.teacher_manager.latest_checkpoint))
    else:
      print("Initializing teacher from scratch.")

  def restore_student(self):
    self.student_ckpt.restore(self.student_manager.latest_checkpoint)
    if self.student_manager.latest_checkpoint:
      print("Restored teacher from {}".format(self.student_manager.latest_checkpoint))
    else:
      print("Initializing teacher from scratch.")

  def distill_loop(self, padding_symbol=0):
    @tf.function(experimental_relax_shapes=True)
    def get_logits(x):
      return self.student_model(x)

    @tf.function(experimental_relax_shapes=True)
    def train_step(x, y, y_true):
      with tf.GradientTape() as tape:
        logits = self.student_model(x)
        tf.print(logits)
        tf.print(y)
        distill_loss = self.student_model.loss(y_pred=logits, y_true=y)

      grads = tape.gradient(distill_loss, self.student_model.trainable_weights)
      self.student_model.optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))
      actual_loss = masked_sequence_loss(y_pred=logits, y_true=y_true)
      return distill_loss, actual_loss

    train_iter = iter(self.task.train_dataset)
    valid_iter = iter(self.task.valid_dataset)

    step = 0
    epochs = 0
    num_epochs = self.distill_params.n_epochs
    for (x, y) in train_iter:
      x = tf.convert_to_tensor(x, dtype=tf.int64)
      y = tf.convert_to_tensor(y, dtype=tf.int64)

      teacher_logits = self.teacher_model(x)
      masked_teacher_probs = get_probs(logits=teacher_logits, labels=y, temperature=self.temperature)

      distill_loss, actual_loss = train_step(x=x, y=masked_teacher_probs, y_true=y)
      # Log every 200 batches.
      if step % 200 == 0:
        log_summary(log_name='learning_rate',
                    log_value=self.student_model.optimizer.learning_rate(self.student_model.optimizer.iterations),
                    summary_scope='train')
        log_summary(log_name='fine_distill_loss', log_value=distill_loss, summary_scope='train')

      if (step % self.task.n_train_batches) == 0:
        self.validate(actual_loss, distill_loss, valid_iter)
        epochs += 1

      step += 1

      if epochs >= num_epochs:
        break

  @tf.function(experimental_relax_shapes=True)
  def validate(self, actual_loss, distill_loss, valid_iter):
    valid_step = 0
    validation_actual_loss = tf.keras.metrics.Mean()
    validation_distill_loss = tf.keras.metrics.Mean()
    while valid_step < self.task.n_valid_batches:
      v_x, v_y = next(valid_iter)
      v_x = tf.convert_to_tensor(v_x, dtype=tf.int64)
      v_y = tf.convert_to_tensor(v_y, dtype=tf.int64)

      teacher_logits = self.teacher_model(v_x)
      masked_teacher_probs = get_probs(logits=teacher_logits,
                                       labels=v_y,
                                       temperature=self.temperature)

      logits = self.student_model(v_x)
      validation_distill_loss.update_state(masked_sequence_loss_with_probs(y_pred=logits,
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

  def run(self):
    raise NotImplementedError
