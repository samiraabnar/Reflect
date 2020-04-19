from util.distill_params import DISTILL_PARAMS
from util.model_configs import GPT2Config, ModelConfig, MODEL_CONFIGS, CapsConfig, ResnetConfig
from util.train_params import TRAIN_PARAMS


class TrainParams(object):
  def __init__(self, optimizer,
               learning_rate=0.0001,
               n_epochs=60,
               warmup_steps=5000,
               decay_steps=10000,
               hold_base_rate_steps=1000,
               total_training_steps=60000,
               num_train_epochs=60,
               decay_rate=0.96,
               schedule='',
  ):
    self.learning_rate = learning_rate
    self.n_epochs = n_epochs
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.hold_base_rate_steps = hold_base_rate_steps
    self.total_training_steps = total_training_steps
    self.num_train_epochs = num_train_epochs
    self.optimizer =  optimizer
    self.schedule = schedule
    self.decay_rate = decay_rate


class DistillParams(object):
  def __init__(self,
               distill_temp=5.0,
               student_distill_rate=0.9,
               student_gold_rate=0.1,
               student_learning_rate=0.0001,
               student_decay_steps=10000,
               student_warmup_steps=10000,
               student_hold_base_rate_steps=1000,
               student_decay_rate=0.96,
               student_optimizer='adam',
               teacher_learning_rate=0.0001,
               teacher_decay_steps=10000,
               teacher_warmup_steps=10000,
               teacher_hold_base_rate_steps=1000,
               teacher_decay_rate=0.96,
               teacher_optimizer='radam',
               n_epochs=60,
               schedule='',
               distill_decay_steps=1000000,
               distill_warmup_steps=0,
               hold_base_distillrate_steps=1000000,
               student_distill_rep_rate=1.0,
               distill_min_rate=0.0,
               distill_schedule='cnst',
  ):
    self.distill_temp = distill_temp
    self.distill_schedule = distill_schedule
    self.student_distill_rate = student_distill_rate
    self.distill_min_rate = distill_min_rate
    self.student_gold_rate = student_gold_rate
    self.student_learning_rate = student_learning_rate
    self.student_decay_steps = student_decay_steps
    self.student_warmup_steps = student_warmup_steps
    self.student_hold_base_rate_steps = student_hold_base_rate_steps
    self.student_optimizer = student_optimizer
    self.teacher_learning_rate = teacher_learning_rate
    self.teacher_warmup_steps = teacher_warmup_steps
    self.teacher_decay_steps = teacher_decay_steps
    self.teacher_optimizer = teacher_optimizer
    self.teacher_hold_base_rate_steps = teacher_hold_base_rate_steps
    self.n_epochs = n_epochs
    self.schedule = schedule
    self.distill_decay_steps = distill_decay_steps
    self.distill_warmup_steps = distill_warmup_steps
    self.hold_base_distillrate_steps = hold_base_distillrate_steps
    self.student_distill_rep_rate = student_distill_rep_rate
    self.teacher_decay_rate = teacher_decay_rate
    self.student_decay_rate = student_decay_rate


class TaskParams:
  def __init__(self, batch_size=64, num_replicas_in_sync=1):
    self.batch_size = batch_size
    self.num_replicas_in_sync = num_replicas_in_sync

def get_train_params(train_config):
  train_params = TrainParams(**TRAIN_PARAMS[train_config])

  return train_params

def get_distill_params(distill_config):
  if distill_config != 'base':
   return DistillParams(**DISTILL_PARAMS[distill_config])

  return DistillParams()


def get_task_params(**kwargs):
  task_params = TaskParams(**kwargs)
  return task_params

def get_model_params(task, config_name='', model_config='base'):
  print("model config:", model_config)
  if model_config in MODEL_CONFIGS:
    model_cnfgs = MODEL_CONFIGS.get(model_config)
  else:
    model_cnfgs = MODEL_CONFIGS.get('base')

  if 'gpt' in config_name or 'bert' in config_name:
    return GPT2Config(vocab_size=task.vocab_size(),
                      output_dim=task.output_size(),
                      num_labels=task.output_size(),
                      **model_cnfgs)
  elif 'caps' in config_name:
    return CapsConfig(output_dim=task.output_size(),
                      **model_cnfgs)
  elif 'resnet' in config_name:
    return ResnetConfig(output_dim=task.output_size(),
                      **model_cnfgs)
  else:
    return ModelConfig(input_dim=task.vocab_size(),
                       output_dim=task.output_size(),**model_cnfgs)



