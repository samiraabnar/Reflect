class Model(object):
  def apply(self, examples):
    raise NotImplementedError

  def update(self, loss):
    raise NotImplementedError