from util.model_configs import GPT2Config, ModelConfig, MODEL_CONFIGS


class TrainParams:
  learning_rate = 0.001
  n_epochs = 60
  warmup_steps = 5000
  decay_steps = 10000
  hold_base_rate_steps = 1000
  total_training_steps = 60000
  num_train_epochs = 60
  optimizer = 'adam'

class DistillParams:
  distill_temp = 1.0
  student_distill_rate = 0.9
  student_gold_rate = 0.1
  student_learning_rate = 0.001
  student_decay_steps = 10000
  student_warmup_steps = 10000
  student_optimizer = 'adam'
  n_epochs = 30

class TaskParams:
  batch_size = 128

def get_train_params():
  train_params = TrainParams()
  return train_params

def get_distill_params():
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
  if config_name == 'lm_gpt2':
    return GPT2Config(vocab_size=task.databuilder.vocab_size(),**model_cnfgs)
  else:
    return ModelConfig(input_dim=task.databuilder.vocab_size(),
                       output_dim=task.databuilder.vocab_size(),**model_cnfgs)


