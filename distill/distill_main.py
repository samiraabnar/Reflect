import tensorflow as tf
from distill.distill_util import DistillLoss
from distill.distiller import Distiller
from util.config_util import get_task_params, get_model_params, get_distill_params
import os
from tasks.tasks import SvAgreementLM, WordSvAgreementLM
from tf2_models.lm_transformer import LmGPT2
from util.config_util import get_model_params, get_task_params, get_train_params
from tf2_models.lm_lstm import LmLSTM
from absl import flags
import sys

FLAGS = flags.FLAGS
flags.DEFINE_string('task', 'word_sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement_lm')


flags.DEFINE_string('teacher_exp_name', 'trial4', 'experiment directory')
flags.DEFINE_string('teacher_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

flags.DEFINE_string('student_exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('student_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

FLAGS(sys.argv)
hparams = flags.FLAGS


MODELS = {"lm_lstm": LmLSTM,
          "lm_gpt2": LmGPT2}

TASKS = {
  'sv_agreement_lm': SvAgreementLM,
  'word_sv_agreement_lm': WordSvAgreementLM,
}

if __name__ == '__main__':
  log_dir = "logs"
  chkpt_dir = "tf_ckpts"


  # Create task
  task = TASKS[hparams.task](get_task_params())

  # Create the Model
  teacher_model = MODELS[hparams.teacher_model](hparams=get_model_params(task, hparams.teacher_model))
  student_model = MODELS[hparams.student_model](hparams=get_model_params(task, hparams.teacher_model))

  teacher_log_dir = os.path.join(log_dir, task.name, teacher_model.model_name + "_" + hparams.teacher_exp_name)
  teacher_ckpt_dir = os.path.join(chkpt_dir, task.name, teacher_model.model_name + "_" + hparams.teacher_exp_name)

  student_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=DistillLoss(tmp=1))

  print(teacher_ckpt_dir)
  teacher_ckpt = tf.train.Checkpoint(net=teacher_model)
  teacher_manager = tf.train.CheckpointManager(teacher_ckpt, teacher_ckpt_dir, max_to_keep=2)
  teacher_ckpt.restore(teacher_manager.latest_checkpoint)
  if teacher_manager.latest_checkpoint:
    print("Restored from {}".format(teacher_manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")


  distiller = Distiller(get_distill_params(), teacher_model, student_model, task)
  distiller.distill_loop()