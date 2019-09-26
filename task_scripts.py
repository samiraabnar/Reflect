import tensorflow as tf
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


  # Create the Task
  task = SvAgreementLM(get_task_params())

  with tf.device('/gpu:0'):
    # Create the Model
    lm_lstm = LmLSTM(hparams=get_model_params(task))

    # Create the Trainer
    trainer = Trainer(model=lm_lstm, task=task, train_params=get_train_params())

    #Train
    trainer.train()