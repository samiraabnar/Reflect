import tensorflow as tf
from tf2_models.caps_util import *

from tf2_models.em_routing import EmRouting


class ConvCaps(tf.keras.layers.Layer):
  def __init__(self, hparams, num_output_caps, kernel, stride, kh_kw_i, scope='conv_caps', *inputs, **kwargs):
    super(ConvCaps, self).__init__(hparams, name=scope, *inputs, **kwargs)
    self.hparams = hparams
    self.num_output_caps = num_output_caps
    self.kernel = kernel
    self.stride = stride
    self.kh_kw_i = kh_kw_i
    self.weights_regularizer = tf.keras.regularizers.l2(self.hparams.l2)
    self.w =  tf.Variable(name='w',
                          initial_value=tf.random.truncated_normal(shape=[1, self.kh_kw_i, self.num_output_caps, 4, 4],
                                                                   dtype=tf.float32))
    self.em_routing = EmRouting(hparams, num_output_caps=num_output_caps)

  def compute_votes(self, poses_i, tag=False):
    """Compute the votes by multiplying input poses by transformation matrix.

    Multiply the poses of layer i by the transform matrix to compute the votes for
    layer j.

    Author:
      Ashley Gritzman 19/10/2018

    Credit:
      Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-Tensorflow"
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow

    Args:
      poses_i:
        poses in layer i tiled according to the kernel
        (N*output_h*output_w, kernel_h*kernel_w*i, 16)
        (64*5*5, 9*8, 16)

    Returns:
      votes:
        (N*output_h*output_w, kernel_h*kernel_w*i, o, 16)
        (64*5*5, 9*8, 32, 16)
    """

    batch_size = tf.shape(poses_i)[0] # 64*5*5
    kh_kw_i = tf.shape(poses_i)[1] # 9*8


    # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 1, 4, 4)
    output = tf.reshape(poses_i, shape=[batch_size, kh_kw_i, 1, 4, 4])

    # the output of capsule is miu, the mean of a Gaussian, and activation, the
    # sum of probabilities it has no relationship with the absolute values of w
    # and votes using weights with bigger stddev helps numerical stability


    # (1, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    w = tf.tile(self.w, [batch_size, 1, 1, 1, 1])

    # (64*5*5, 9*8, 1, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    output = tf.tile(output, [1, 1, self.num_output_caps, 1, 1])

    # (64*5*5, 9*8, 32, 4, 4) x (64*5*5, 9*8, 32, 4, 4)
    # -> (64*5*5, 9*8, 32, 4, 4)
    mult = tf.matmul(output, w)

    # (64*5*5, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 16)
    votes = tf.reshape(mult, [batch_size, kh_kw_i, self.num_output_caps, 16])

    # tf.summary.histogram('w', w)

    return votes

  def call(self, inputs_pose, inputs_activation, training=False, **kwargs):

    # Get shapes
    shape = tf.shape(inputs_pose)
    batch_size = shape[0]
    child_space = shape[1]
    child_space_2 = tf.cast(child_space ** 2, tf.int32)
    child_caps = shape[3]
    parent_space = tf.cast(tf.floor((child_space - self.kernel) / self.stride + 1), tf.int32)
    parent_space_2 = tf.cast(parent_space ** 2, tf.int32)
    parent_caps = self.num_output_caps
    kernel_2 = tf.cast(self.kernel ** 2, tf.int32)

    # Votes
    # Tile poses and activations
    # (64, 7, 7, 8, 16)  -> (64, 5, 5, 9, 8, 16)
    pose_tiled, spatial_routing_matrix = kernel_tile(
      inputs_pose,
      kernel=self.kernel,
      stride=self.stride)
    activation_tiled, _ = kernel_tile(
      inputs_activation,
      kernel=self.kernel,
      stride=self.stride)

    # Check dimensions of spatial_routing_matrix
    # assert [a for a in tf.shape(spatial_routing_matrix)] == [child_space_2, parent_space_2]

    # Unroll along batch_size and parent_space_2
    # (64, 5, 5, 9, 8, 16) -> (64*5*5, 9*8, 16)
    pose_unroll = tf.reshape(
      pose_tiled,
      shape=[batch_size * parent_space_2, kernel_2 * child_caps, 16])
    activation_unroll = tf.reshape(
      activation_tiled,
      shape=[batch_size * parent_space_2, kernel_2 * child_caps, 1])

    # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 32, 16)
    votes = self.compute_votes(
      pose_unroll,
      tag=True)

    # Routing
    # votes (64*5*5, 9*8, 32, 16)
    # activations (64*5*5, 9*8, 1)
    # pose_out: (N, output_h, output_w, o, 4x4)
    # activation_out: (N, output_h, output_w, o, 1)
    pose_out, activation_out = self.em_routing(votes,
                                               activation_unroll,
                                               batch_size,
                                               spatial_routing_matrix)

    return pose_out, activation_out


class FcCaps(tf.keras.layers.Layer):
  def __init__(self, hparams, scope='class_caps', *inputs, **kwargs):
    super(FcCaps, self).__init__(hparams, name=scope, *inputs, **kwargs)
    self.hparams  = hparams
    self.num_output_caps = self.hparams.output_dim
    self.kh_kw_i = self.hparams.D
    self.w = tf.Variable(name='w',
                         initial_value=tf.random.truncated_normal(shape=[1, self.kh_kw_i, self.num_output_caps, 4, 4],
                                                                  dtype=tf.float32))
    self.em_routing = EmRouting(hparams, num_output_caps=self.num_output_caps)


  def call(self,pose_in, activation_in, training=True, **kwargs):
    """Fully connected capsule layer.

    "The last layer of convolutional capsules is connected to the final capsule
    layer which has one capsule per output class." We call this layer 'fully
    connected' because it fits these characteristics, although Hinton et al. do
    not use this teminology in the paper.

    See Hinton et al. "Matrix Capsules with EM Routing" for detailed description.

    Author:
      Ashley Gritzman 27/11/2018

    Args:
      activation_in:
        (batch_size, child_space, child_space, child_caps, 1)
        (64, 7, 7, 8, 1)
      pose_in:
        (batch_size, child_space, child_space, child_caps, 16)
        (64, 7, 7, 8, 16)
      ncaps_out: number of class capsules
      name:
      weights_regularizer:

    Returns:
      activation_out:
        score for each output class
        (batch_size, ncaps_out)
        (64, 5)
      pose_out:
        pose for each output class capsule
        (batch_size, ncaps_out, 16)
        (64, 5, 16)
    """

    # Get shapes
    shape = tf.shape(pose_in)
    batch_size = shape[0]
    child_space = shape[1]
    child_caps = shape[3]


    # In the class_caps layer, we apply same multiplication to every spatial
    # location, so we unroll along the batch and spatial dimensions
    # (64, 5, 5, 32, 16) -> (64*5*5, 32, 16)
    pose = tf.reshape(
        pose_in,
        shape=[batch_size * child_space * child_space, child_caps, 16])
    activation = tf.reshape(
        activation_in,
        shape=[batch_size * child_space * child_space, child_caps, 1],
        name="activation")

    # (64*5*5, 32, 16) -> (65*5*5, 32, 5, 16)
    votes = self.compute_votes(pose)

    # (65*5*5, 32, 5, 16)
    # assert (
    #   [a for a in tf.shape(votes)] ==
    #   [batch_size * child_space * child_space, child_caps, self.num_output_caps, 16])
    #

    # (64*5*5, 32, 5, 16)
    votes = tf.reshape(
        votes,
        [batch_size, child_space, child_space, child_caps, self.num_output_caps,
         votes.shape[-1]])
    votes = coord_addition(votes)

    # Flatten the votes:
    # Combine the 4 x 4 spacial dimensions to appear as one spacial dimension
    # with many capsules.
    # [64*5*5, 16, 5, 16] -> [64, 5*5*16, 5, 16]
    votes_flat = tf.reshape(
        votes,
        shape=[batch_size, child_space * child_space * child_caps,
               self.num_output_caps, votes.shape[-1]])
    activation_flat = tf.reshape(
        activation,
        shape=[batch_size, child_space * child_space * child_caps, 1])

    spatial_routing_matrix = create_routing_map(child_space=1, k=1, s=1)



    pose_out, activation_out = self.em_routing(votes_flat,
                         activation_flat,
                         batch_size,
                         spatial_routing_matrix)

    activation_out = tf.squeeze(activation_out, name="activation_out")
    pose_out = tf.squeeze(pose_out, name="pose_out")

    return pose_out, activation_out

  def compute_votes(self, poses_i, tag=False):
    """Compute the votes by multiplying input poses by transformation matrix.

    Multiply the poses of layer i by the transform matrix to compute the votes for
    layer j.

    Author:
      Ashley Gritzman 19/10/2018

    Credit:
      Suofei Zhang's implementation on GitHub, "Matrix-Capsules-EM-Tensorflow"
      https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow

    Args:
      poses_i:
        poses in layer i tiled according to the kernel
        (N*output_h*output_w, kernel_h*kernel_w*i, 16)
        (64*5*5, 9*8, 16)

    Returns:
      votes:
        (N*output_h*output_w, kernel_h*kernel_w*i, o, 16)
        (64*5*5, 9*8, 32, 16)
    """

    batch_size = tf.shape(poses_i)[0] # 64*5*5
    kh_kw_i = tf.shape(poses_i)[1] # 9*8

    # (64*5*5, 9*8, 16) -> (64*5*5, 9*8, 1, 4, 4)
    output = tf.reshape(poses_i, shape=[batch_size, kh_kw_i, 1, 4, 4])

    # the output of capsule is miu, the mean of a Gaussian, and activation, the
    # sum of probabilities it has no relationship with the absolute values of w
    # and votes using weights with bigger stddev helps numerical stability


    # (1, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    w = tf.tile(self.w, [batch_size, 1, 1, 1, 1])

    # (64*5*5, 9*8, 1, 4, 4) -> (64*5*5, 9*8, 32, 4, 4)
    output = tf.tile(output, [1, 1, self.num_output_caps, 1, 1])

    # (64*5*5, 9*8, 32, 4, 4) x (64*5*5, 9*8, 32, 4, 4)
    # -> (64*5*5, 9*8, 32, 4, 4)
    mult = tf.matmul(output, w)

    # (64*5*5, 9*8, 32, 4, 4) -> (64*5*5, 9*8, 32, 16)
    votes = tf.reshape(mult, [batch_size, kh_kw_i, self.num_output_caps, 16])

    # tf.summary.histogram('w', w)

    return votes

def coord_addition(votes):
  """Coordinate addition for connecting the last convolutional capsule layer to   the final layer.

  "When connecting the last convolutional capsule layer to the final layer we do
  not want to throw away information about the location of the convolutional
  capsules but we also want to make use of the fact that all capsules of the
  same type are extracting the same entity at different positions. We therefore   share the transformation matrices between different positions of the same
  capsule type and add the scaled coordinate (row, column) of the center of the   receptive field of each capsule to the first two elements of the right-hand
  column of its vote matrix. We refer to this technique as Coordinate Addition.   This should encourage the shared final transformations to produce values for
  those two elements that represent the fine position of the entity relative to   the center of the capsule’s receptive field."

  In Suofei's implementation, they add x and y coordinates as two new dimensions   to the pose matrix i.e. from 16 to 18 dimensions. The paper seems to say that   the values are added to existing dimensions.

  See Hinton et al. "Matrix Capsules with EM Routing" for detailed description
  coordinate addition.

  Author:
    Ashley Gritzman 27/11/2018

  Credit:
    Based on Jonathan Hui's implementation:
    https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-
    Capsule-Network/

  Args:
    votes:
      (batch_size, child_space, child_space, child_caps, n_output_capsules, 16)
      (64, 5, 5, 32, 5, 16)

  Returns:
    votes:
      same size as input, with coordinate encoding added to first two elements
      of right hand column of vote matrix
      (batch_size, parent_space, parent_space, parent_caps, 1)
      (64, 5, 5, 32, 16)
  """

  # get spacial dimension of votes
  vote_shape = tf.shape(votes)
  height = vote_shape[1]
  width = vote_shape[2]
  dims = vote_shape[-1]

  # Generate offset coordinates
  # The difference here is that the coordinate won't be exactly in the middle of
  # the receptive field, but will be evenly spread out
  w_offset_vals = (np.arange(width) + 0.50)/float(width)
  h_offset_vals = (np.arange(height) + 0.50)/float(height)

  w_offset = np.zeros([width, dims]) # (5, 16)
  w_offset[:,3] = w_offset_vals
  # (1, 1, 5, 1, 1, 16)
  w_offset = np.reshape(w_offset, [1, 1, width, 1, 1, dims])

  h_offset = np.zeros([height, dims])
  h_offset[:,7] = h_offset_vals
  # (1, 5, 1, 1, 1, 16)
  h_offset = np.reshape(h_offset, [1, height, 1, 1, 1, dims])

  # Combine w and h offsets using broadcasting
  # w is (1, 1, 5, 1, 1, 16)
  # h is (1, 5, 1, 1, 1, 16)
  # together (1, 5, 5, 1, 1, 16)
  offset = w_offset + h_offset

  # Convent from numpy to tensor
  offset = tf.constant(offset, dtype=tf.float32)

  votes = tf.add(votes, offset, name="votes_with_coord_add")

  return votes
