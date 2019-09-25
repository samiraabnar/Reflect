class Distiller(object):
  def __init__(self, distill_params, teacher_model, student_model, task):
    self.teacher_model = teacher_model
    self.student_model = student_model
    self.task = task
    self.hparams = distill_params

  def get_loss(self, logits, labels, softmax_temp=1.0):
    pass

  def distill_loop(self):

    # Load Data
    examples = self.task.get_examples()

    # Apply teacher and student
    teacher_output_dic = self.teacher_model.apply(examples)
    student_output_dic = self.student_model.apply(teacher_output_dic['inputs'])

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