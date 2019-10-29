from distill.distiller import Distiller
from distill.online_distiller import OnlineDistiller
from util import constants
from util.config_util import get_distill_params, TASKS
import os
from util.config_util import get_model_params, get_task_params, get_train_params
from absl import flags
import sys

from util.model_configs import MODELS

FLAGS = flags.FLAGS
flags.DEFINE_string('log_dir', 'logs', 'log dir')
flags.DEFINE_string('chkpt_dir', 'tf_ckpts', 'checkpoint dir')

flags.DEFINE_string('task', 'word_sv_agreement_lm', 'sv_agreement_lm | word_sv_agreement_lm')
flags.DEFINE_string('distill_config', 'base', ' distillation hparams set')

flags.DEFINE_string('teacher_exp_name', 'trial4', 'experiment directory')
flags.DEFINE_string('teacher_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

flags.DEFINE_string('student_exp_name', 'trial1', 'experiment directory')
flags.DEFINE_string('student_model', 'lm_lstm', 'lm_lstm | lm_gpt2')

flags.DEFINE_string('student_config', 'base', 'base | small_lstm ')
flags.DEFINE_string('teacher_config', 'base', 'base | small_lstm ')

flags.DEFINE_string('distill_mode', 'offline', 'offline | online ')


FLAGS(sys.argv)
hparams = flags.FLAGS


def create_and_load_models():
  cl_token = task.databuilder.sentence_encoder().encode(constants.bos)
  teacher_model = MODELS[hparams.teacher_model](
    hparams=get_model_params(task, hparams.teacher_model, hparams.teacher_config), cl_token=cl_token)
  student_model = MODELS[hparams.student_model](
    hparams=get_model_params(task, hparams.student_model, hparams.student_config), cl_token=cl_token)
  teacher_log_dir = os.path.join(hparams.log_dir, task.name,
                                 "teacher_" + teacher_model.model_name + "_" + hparams.teacher_exp_name)
  teacher_ckpt_dir = os.path.join(hparams.chkpt_dir, task.name,
                                  teacher_model.model_name + "_" + hparams.teacher_exp_name)
  student_log_dir = os.path.join(hparams.log_dir, task.name,
                                 "student_" + student_model.model_name + "_" + hparams.student_exp_name)
  student_ckpt_dir = os.path.join(hparams.chkpt_dir, task.name,
                                  "student_" + student_model.model_name + "_" + hparams.student_exp_name)

  return teacher_model, student_model, teacher_log_dir, teacher_ckpt_dir, student_log_dir, student_ckpt_dir

DISTILLER = {'offline': Distiller,
             'online': OnlineDistiller}

if __name__ == '__main__':

  # Create task
  task = TASKS[hparams.task](get_task_params())

  # Create the Model
  teacher_model, student_model, \
  teacher_log_dir, teacher_ckpt_dir, student_log_dir, student_ckpt_dir = create_and_load_models()

  distiller = DISTILLER[hparams.distill_mode](distill_params=get_distill_params(hparams.distill_config),
                                              teacher_model=teacher_model,
                                              student_model=student_model,
                                              task=task,
                                              teacher_ckpt_dir=teacher_ckpt_dir,
                                              teacher_log_dir=teacher_log_dir,
                                              student_ckpt_dir=student_ckpt_dir,
                                              student_log_dir=student_log_dir,
                                              )

  # Restore Models
  distiller.restore_teacher()
  distiller.restore_student()

  # Run the distillation loop
  distiller.distill_loop()