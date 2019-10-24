from util.model_configs import GPT2Config, ModelConfig, MODEL_CONFIGS


class TrainParams(object):
  def __init__(self, optimizer,
               learning_rate=0.0001,
               n_epochs=60,
               warmup_steps=5000,
               decay_steps=10000,
               hold_base_rate_steps=1000,
               total_training_steps=60000,
               num_train_epochs=60,
  ):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.hold_base_rate_steps = hold_base_rate_steps
    self.total_training_steps = total_training_steps
    self.num_train_epochs = num_train_epochs
    self.optimizer =  optimizer


class DistillParams:
  distill_temp = 5.0
  student_distill_rate = 0.9
  student_gold_rate = 0.1
  student_learning_rate = 0.0001
  student_decay_steps = 10000
  student_warmup_steps = 10000
  student_optimizer = 'adam'
  n_epochs = 60

class TaskParams:
  batch_size = 64

def get_train_params(train_config):
  train_params = TrainParams(**TRAIN_PARAMS[train_config])

  return train_params

def get_distill_params(distill_config):
  return DistillParams()

def get_task_params():
  task_params = TaskParams()
  return task_params

def get_model_params(task, config_name='', model_config='base'):
  print("model config:", model_config)
  if model_config in MODEL_CONFIGS:
    model_cnfgs = MODEL_CONFIGS.get(model_config)
  else:
    model_cnfgs = MODEL_CONFIGS.get('base')

  print(model_cnfgs)
  if 'gpt' in config_name:
    return GPT2Config(vocab_size=task.vocab_size(),
                      output_dim=task.output_size(),
                      num_labels=task.output_size(),
                      **model_cnfgs)
  else:
    return ModelConfig(input_dim=task.vocab_size(),
                       output_dim=task.output_size(),**model_cnfgs)


radam_slow = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 0
}

adam_slow = {
'learning_rate': 0.0001,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}


adam_mid = {
'learning_rate': 0.0005,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}

adam_midmid = {
'learning_rate': 0.0002,
'optimizer': 'adam',
'hold_base_rate_steps': 0
}

radam_fast_long = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}

radam_slow_long = {
'learning_rate': 0.0001,
'optimizer': 'radam',
'hold_base_rate_steps': 1000000
}


radam_fast = {
'learning_rate': 0.001,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}


radam_mid = {
'learning_rate': 0.0005,
'optimizer': 'radam',
'hold_base_rate_steps': 10000
}

TRAIN_PARAMS = {'radam_slow': radam_slow,
                'radam_fast': radam_fast,
                'adam_slow':  adam_slow,
                'radam_fast_long': radam_fast_long,
                'radam_slow_long': radam_slow_long,
                'adam_mid': adam_mid,
                'adam_midmid': adam_midmid,
                'radam_mid': radam_mid}
