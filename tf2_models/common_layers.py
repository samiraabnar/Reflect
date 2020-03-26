import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest


def gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
      x: float Tensor to perform activation.
  Returns:
      `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
    (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

def shape_list(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def get_initializer(initializer_range=0.02):
  """Creates a `tf.initializers.truncated_normal` with the given range.
  Args:
    initializer_range: float, initializer range for stddev.
  Returns:
    TruncatedNormal initializer with stddev = `initializer_range`.
  """
  return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

def make_variable_state_initializer(**kwargs):
  def variable_state_initializer(shape, batch_size, dtype, index):
    args = kwargs.copy()

    if args.get('name'):
      args['name'] = args['name'] + '_' + str(index)
    else:
      args['name'] = 'init_state_' + str(index)

    args['shape'] = shape
    args['dtype'] = dtype

    var = tf.get_variable(**args)
    var = tf.expand_dims(var, 0)
    var = tf.tile(var, tf.pack([batch_size] + [1] * len(shape)))
    var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return var

  return variable_state_initializer

def get_initial_cell_state(cell, initializer, batch_size, dtype):
  """Return state tensor(s), initialized with initializer.
  Args:
    cell: RNNCell.
    batch_size: int, float, or unit Tensor representing the batch size.
    initializer: function with two arguments, shape and dtype, that
        determines how the state is initialized.
    dtype: the data type to use for the state.
  Returns:
    If `state_size` is an int or TensorShape, then the return value is a
    `N-D` tensor of shape `[batch_size x state_size]` initialized
    according to the initializer.
    If `state_size` is a nested list or tuple, then the return value is
    a nested list or tuple (of the same structure) of `2-D` tensors with
  the shapes `[batch_size x s]` for each s in `state_size`.
  """
  state_size = cell.state_size
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    init_state_flat = [
      initializer(s, batch_size, dtype, i)
      for i, s in enumerate(state_size_flat)]
    init_state = nest.pack_sequence_as(structure=state_size,
                                       flat_sequence=init_state_flat)
  else:
    init_state_size = state_size
    init_state = initializer(init_state_size, batch_size, dtype, None)

  return init_state

def _generate_variable_state(batch_size_tensor, state_size, dtype):
  """Generate a variable tensor with shape [batch_size, state_size]."""
  def create_variable(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return tf.Variable(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_variable, state_size)
  else:
    return create_variable(state_size)
