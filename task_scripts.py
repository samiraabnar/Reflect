from pipeline.tasks import SvAgreementLM
from pipeline.trainer import Trainer
from tf2_models.lm_lstm import LmLSTM


def get_task_params():
  class task_params(object):
    batch_size = 64

  return task_params

def get_model_params(task):
  class model_params(object):
    hidden_dim=32
    input_dim=task.databuilder.vocab_size()
    output_dim=task.databuilder.vocab_size()
    depth=2
    hidden_dropout_rate=0.1

  return model_params

if __name__ == '__main__':

  # Create the Task
  task = SvAgreementLM(get_task_params())

  # Create the Model
  lm_lstm = LmLSTM(hparams=get_model_params(task))

  # Create the Trainer
  trainer = Trainer(model=lm_lstm, task=task, train_params=None)
  trainer.train()