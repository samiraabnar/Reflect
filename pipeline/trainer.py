import tensorflow as tf
import os
import numpy as np
from pipeline import  train_utils
from pipeline.train_utils import cosine_decay_with_warmup


class Trainer(object):
  def __init__(self, model, task, train_params):
    self.n_epochs = train_params.n_epochs
    self.learning_rate_base = train_params.learning_rate
    self.total_training_steps = train_params.total_training_steps
    self.train_params = train_params
    self.model = model
    self.task = task

    with tf.device('/gpu:0'):
      self.learning_rate = cosine_decay_with_warmup(tf.compat.v1.train.get_or_create_global_step(),
                                                    self.learning_rate_base,
                                                    total_steps=self.total_training_steps,
                                                    warmup_learning_rate=0.0,
                                                    warmup_steps=train_params.warmpup_steps,
                                                    hold_base_rate_steps=train_params.hold_base_rate_steps)

      self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate,
                                          beta_1=0.9,
                                          beta_2=0.999,
                                          epsilon=1e-07,
                                          amsgrad=False,
                                          name='Adam')
      self.optimizer.iterations = tf.compat.v1.train.get_or_create_global_step()

    with tf.device('/cpu:0'):
      summary_path = os.path.join('logs', self.task.name, self.model.model_name)

      self.train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_path, 'train'))
      self.eval_summary_writer = tf.summary.create_file_writer(os.path.join(summary_path, 'eval'))

  @tf.function
  def train_step(self):


    t_loss = train_utils.train(self.model, self.task.train_dataset,
                               self.optimizer, self.task.get_loss_fn(),
                               avg_metric_dic=self.train_avg_metric_dic, task=self.task)
    with self.train_summary_writer.as_default():
      for metric in self.train_avg_metric_dic:
        tf.summary.scalar('loss', self.train_avg_metric_dic[metric].result(), step=self.optimizer.iterations)
        self.train_avg_metric_dic[metric].reset_states()
    train_utils.eval(self.model,
                     self.task.valid_dataset,
                     avg_metric_dic=self.eval_avg_metric_dic, task=self.task, step_num=self.optimizer.iterations)

    with self.eval_summary_writer.as_default():
      for metric in self.eval_avg_metric_dic:
        tf.summary.scalar('loss', self.eval_avg_metric_dic[metric].result(), step=self.optimizer.iterations)
        self.eval_avg_metric_dic[metric].reset_states()

    return t_loss

  def train(self, ckpt, manager):
    print("Start training ... ")
    print("initial learning rate is ", self.optimizer.learning_rate)

    self.train_avg_metric_dic = {}
    for metric in self.task.metrics:
      self.train_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)

    self.eval_avg_metric_dic = {}
    for metric in self.task.metrics:
      print(metric)
      self.eval_avg_metric_dic[metric] = tf.keras.metrics.Mean(name=metric, dtype=tf.float32)

    for i in range(self.n_epochs):
      t_loss = self.train_step()

      ckpt.step.assign_add(1)
      if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss@epoch%d:" % i, t_loss)



