import tensorflow as tf

class Distiller(object):
  def __init__(self, distill_params, teacher_model, student_model, task):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.hparams = distill_params

  def get_loss(self, logits, labels, softmax_temp=1.0):


  def distill_loop(self, padding_symbol=0):

    # Load Data
    examples = self.task.get_examples()

    # Apply teacher and student
    for x,y in self.task.train_data:
      teacher_logits = self.teacher_model(x)
      teacher_probs = tf.nn.softmax(teacher_logits/self.hparams.distill_temp, axis=-1)
      sequence_mask = tf.cast(y != 0, dtype=tf.float32)
      teacher_probs = teacher_probs * sequence_mask[...,None] + tf.eye(teacher_logits.shape[-1])[0] * (1 - sequence_mask[...,None])


      student_output_dic = self.student_model(teacher_output_dic['inputs'])

    # Compute Loss for the student
    student_distill_loss = self.get_loss(logits=student_output_dic['logits'],
                                                 labels=teacher_output_dic['logits'],
                                                 softmax_temp=self.hparams.distill_temp)
    student_gold_loss = self.get_loss(logits=student_output_dic['logits'],
                                      labels=student_output_dic['logits'])

    final_loss = self.hparams.student_distill_rate  * student_distill_loss + \
                 self.hparams.student_gold_rate * student_gold_loss

    self.student_model.update(final_loss)

  def run(self):
    raise NotImplementedError
