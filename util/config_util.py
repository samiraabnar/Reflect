class TrainParams:
  learning_rate = 0.001
  n_epochs = 30
  warmpup_steps = 10000
  hold_base_rate_steps = 1000
  total_training_steps = 60000
  num_train_epochs = 30

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