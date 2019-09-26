import tensorflow as tf
import numpy as np
import os
from pipeline.tasks import SvAgreementLM
from pipeline.trainer import Trainer
from tf2_models.lm_lstm import LmLSTM


class TrainParams:
  learning_rate = 0.001
  n_epochs = 30
  warmpup_steps = 10000
  hold_base_rate_steps = 1000
  total_training_steps = 60000


class TaskParams:
  batch_size = 128


class ModelParams:
  hidden_dim=256
  input_dim=None
  output_dim=None
  depth=2
  hidden_dropout_rate=0.5
  input_dropout_rate=0.2


def get_train_params():
  train_params = TrainParams()
  return train_params

def get_task_params():
  task_params = TaskParams()
  return task_params

def get_model_params(task):
  model_params = ModelParams()
  model_params.input_dim = task.databuilder.vocab_size()
  model_params.output_dim = task.databuilder.vocab_size()
  return model_params

if __name__ == '__main__':
  # tf.debugging.set_log_device_placement(True)
  tf.enable_v2_behavior()



  # Create the Task
  task = SvAgreementLM(get_task_params())
  for x,y in task.train_dataset:
    print("Data shape:", x.shape)

  # Create the Model
  model = LmLSTM(hparams=get_model_params(task))



  # # Instantiate an optimizer to train the model.
  # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
  # # Instantiate a loss function.
  # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
  #
  #
  #
  #
  # model.compile(loss=loss_fn, optimizer=optimizer)
  # history = model.fit(task.train_dataset, epochs=5)

  # Create the Trainer
  trainer = Trainer(model=model, task=task, train_params=get_train_params())

  # #Train
  ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, net=trainer.model)
  chpt_path = os.path.join('tf_ckpts', trainer.task.name, trainer.model.model_name)
  manager = tf.train.CheckpointManager(ckpt,
                                       chpt_path,
                                       max_to_keep=3)
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")
    print("Saving params")
    if not os.path.exists(chpt_path):
      os.makedirs(chpt_path)
    np.save(os.path.join(chpt_path, 'task_params'), task.task_params)
    np.save(os.path.join(chpt_path, 'model_params'), model.hparams)
    np.save(os.path.join(chpt_path, 'train_params'), trainer.train_params)

    trainer.train(ckpt, manager)