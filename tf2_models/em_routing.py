import tensorflow as tf
from tf2_models.caps_util import *

def init_rr(spatial_routing_matrix, child_caps, parent_caps):
  """Initialise routing weights.

  Initialise routing weights taking into accout spatial position of child
  capsules. Child capsules in the corners only go to one parent capsule, while
  those in the middle can go to kernel*kernel capsules.

  Author:
    Ashley Gritzman 19/10/2018

  Args:
    spatial_routing_matrix:
      A 2D numpy matrix containing mapping between children capsules along the
      rows, and parent capsules along the columns.
      (child_space^2, parent_space^2)
      (7*7, 5*5)
    child_caps: number of child capsules along depth dimension
    parent_caps: number of parent capsules along depth dimension

  Returns:
    rr_initial:
      initial routing weights
      (1, parent_space, parent_space, kk, child_caps, parent_caps)
      (1, 5, 5, 9, 8, 32)
  """

  # Get spatial dimension of parent & child
  parent_space_2 = tf.cast(spatial_routing_matrix.shape[1], tf.int32)
  parent_space = tf.cast(tf.math.sqrt(tf.cast(parent_space_2, dtype=tf.float32)), tf.int32)
  child_space_2 = tf.cast(spatial_routing_matrix.shape[0], tf.int32)
  child_space = tf.cast(tf.math.sqrt(tf.cast(child_space_2, dtype=tf.float32)), tf.int32)

  # Count the number of parents that each child belongs to
  parents_per_child = tf.reduce_sum(spatial_routing_matrix, axis=1, keepdims=True)

  # Divide the vote of each child by the number of parents that it belongs to
  # If the striding causes the filter not to fit, it will result in some
  # "dropped" child capsules, which effectively means child capsules that do not
  # have any parents. This would create a divide by 0 scenario, so need to add
  # 1e-9 to prevent NaNs.
  tf.print('rr_init')
  tf.print(parents_per_child)
  tf.print(parent_caps)
  tf.print(spatial_routing_matrix)
  rr_initial = (spatial_routing_matrix
                / (tf.cast(parents_per_child, dtype=tf.float32) * tf.cast(parent_caps, dtype=tf.float32) + 1e-9))

  # Convert the sparse matrix to be compatible with votes.
  # This is done by selecting the child capsules belonging to each parent, which
  # is achieved by selecting the non-zero values down each column. Need the
  # combination of two transposes so that order is correct when reshaping
  mask = spatial_routing_matrix.astype(bool)

  tf.print(rr_initial)
  tf.print(rr_initial.shape)
  rr_initial = tf.transpose(rr_initial)[tf.transpose(mask)]
  rr_initial = tf.reshape(rr_initial, [parent_space, parent_space, -1])
  print(rr_initial.shape)
  # Copy values across depth dimensions
  # i.e. the number of child_caps and the number of parent_caps
  # (5, 5, 9) -> (5, 5, 9, 8, 32)
  rr_initial = rr_initial[..., tf.newaxis, tf.newaxis]
  rr_initial = tf.tile(rr_initial, [1, 1, 1, child_caps, parent_caps])

  # Add one mode dimension for batch size
  rr_initial = tf.expand_dims(rr_initial, 0)

  # Check the total of the routing weights is equal to the number of child
  # capsules
  # child_space * child_space * child_caps (minus the dropped ones)
  dropped_child_caps = tf.reduce_sum(tf.cast(tf.reduce_sum(spatial_routing_matrix, axis=1) < 1e-9, dtype=tf.int32))
  effective_child_cap = ((child_space * child_space - dropped_child_caps)
                         * child_caps)

  sum_routing_weights = tf.reduce_sum(rr_initial)


  assert tf.abs(sum_routing_weights - tf.cast(effective_child_cap, dtype=tf.float32)) < 1e-3

  return rr_initial


class EmRouting(tf.keras.layers.Layer):

  def __init__(self, hparams, num_output_caps, scope='em_routing', *inputs, **kwargs):
    super(EmRouting, self).__init__(hparams, name=scope, *inputs, **kwargs)
    self.hparams = hparams
    self.num_out_caps = num_output_caps

    self.beta_a = tf.Variable(
      name='beta_a',
      initial_value=tf.random.truncated_normal(shape=[1, 1, 1, 1, self.num_out_caps, 1],
      dtype=tf.float32))

    # One beta per output capsule type
    # (1, 1, 1, 1, 32, 1)
    # (N, output_h, output_h, num_input_caps, o, n_channels)
    self.beta_v = tf.Variable(
      name='beta_v',
      initial_value=tf.random.truncated_normal(shape=[1, 1, 1, 1, self.num_out_caps, 1],
                                               dtype=tf.float32))



  def call(self,votes_ij, activations_i, batch_size, spatial_routing_matrix):
    """The EM routing between input capsules (i) and output capsules (j).

    See Hinton et al. "Matrix Capsules with EM Routing" for detailed description
    of EM routing.

    Author:
      Ashley Gritzman 19/10/2018
    Definitions:
      N -> number of samples in batch
      output_h -> output height
      output_w -> output width
      kernel_h -> kernel height
      kernel_w -> kernel width
      kk -> kernel_h * kernel_w
      num_input_caps -> number of input capsules, also called "child_caps"
      o -> number of output capsules, also called "parent_caps"
      child_space -> spatial dimensions of input capsule layer i
      parent_space -> spatial dimensions of output capsule layer j
      n_channels -> number of channels in pose matrix (usually 4x4=16)
    Args:
      votes_ij:
        votes from capsules in layer i to capsules in layer j
        For conv layer:
          (N*output_h*output_w, kernel_h*kernel_w*num_input_caps, o, 4x4)
          (64*6*6, 9*8, 32, 16)
        For FC layer:
          The kernel dimensions are equal to the spatial dimensions of the input
          layer i, and the spatial dimensions of the output layer j are 1x1.
          (N*1*1, child_space*child_space*num_input_caps, o, 4x4)
          (64, 4*4*16, 5, 16)
      activations_i:
        activations of capsules in layer i (L)
        (N*output_h*output_w, kernel_h*kernel_w*num_input_caps, 1)
        (64*6*6, 9*8, 1)
      batch_size:
      spatial_routing_matrix:
    Returns:
      poses_j:
        poses of capsules in layer j (L+1)
        (N, output_h, output_w, o, 4x4)
        (64, 6, 6, 32, 16)
      activations_j:
        activations of capsules in layer j (L+1)
        (N, output_h, output_w, o, 1)
        (64, 6, 6, 32, 1)
    """

    # ----- Dimensions -----#

    # Get dimensions needed to do conversions
    N = batch_size
    votes_shape = votes_ij.get_shape().as_list()
    output_h = tf.math.sqrt(int(votes_shape[0]) / N)
    output_h = tf.cast(output_h, dtype=tf.int32)
    output_w = tf.math.sqrt(int(votes_shape[0]) / N)
    output_w = tf.cast(output_w, tf.int32)
    kh_kw_i = tf.cast(votes_shape[1], tf.int32)
    num_ouput_caps = tf.cast(votes_shape[2], tf.int32)
    n_channels = tf.cast(votes_shape[3], tf.int32)

    # Calculate kernel size by adding up column of spatial routing matrix
    # Do this before conventing the spatial_routing_matrix to tf
    tf.print(spatial_routing_matrix[:, 0].shape)
    kk =tf.cast(tf.reduce_sum(spatial_routing_matrix[:, 0]), tf.int32)

    parent_caps = num_ouput_caps
    child_caps = tf.cast(kh_kw_i / kk, dtype=tf.int32)

    rt_mat_shape = spatial_routing_matrix.shape
    child_space_2 = tf.cast(rt_mat_shape[0], dtype=tf.int32)
    child_space = tf.cast(tf.math.sqrt(tf.cast(child_space_2, dtype=tf.float32)), tf.int32)
    parent_space_2 = tf.cast(rt_mat_shape[1], dtype=tf.int32)
    parent_space = tf.cast(tf.math.sqrt(tf.cast(parent_space_2, dtype=tf.float32)), tf.int32)

    # ----- Reshape Inputs -----#
    # conv: (N*output_h*output_w, kernel_h*kernel_w*num_input_caps, o, 4x4) -> (N, output_h, output_w, kernel_h*kernel_w*num_input_caps, o, 4x4)
    # FC: (N, child_space*child_space*num_input_caps, o, 4x4) -> (N, 1, 1, child_space*child_space*num_input_caps, output_classes, 4x4)
    votes_ij = tf.reshape(votes_ij, [N, output_h, output_w, kh_kw_i, num_ouput_caps, n_channels])

    # (N*output_h*output_w, kernel_h*kernel_w*num_input_caps, 1) -> (N, output_h, output_w, kernel_h*kernel_w*num_input_caps, o, n_channels)
    #              (24, 6, 6, 288, 1, 1)
    activations_i = tf.reshape(activations_i, [N, output_h, output_w, kh_kw_i, 1, 1])

    # Initialise routing assignments
    # rr (1, 6, 6, 9, 8, 16)
    #  (1, parent_space, parent_space, kk, child_caps, parent_caps)
    rr = init_rr(spatial_routing_matrix, child_caps,
                 parent_caps)

    # Need to reshape (1, 6, 6, 9, 8, 16) -> (1, 6, 6, 9*8, 16, 1)
    rr = tf.reshape(
      rr,
      [1, parent_space, parent_space, kk * child_caps, parent_caps, 1])

    # Convert rr from np to tf
    rr = tf.constant(rr, dtype=tf.float32)

    for it in range(self.hparams.iter_routing):
      final_lambda = self.hparams.final_lambda
      inverse_temperature = (final_lambda *
                             (1 - tf.math.pow(0.95, tf.cast(it + 1, tf.float32))))

      # AG 26/06/2018: added var_j
      activations_j, mean_j, var_j = self.m_step(
        rr,
        votes_ij,
        activations_i,
        self.beta_v, self.beta_a,
        inverse_temperature=inverse_temperature)

      # We skip the e_step call in the last iteration because we only need to
      # return the a_j and the mean from the m_stp in the last iteration to
      # compute the output capsule activation and pose matrices
      if it < self.hparams.iter_routing - 1:
        rr = self.e_step(votes_ij,
                    activations_j,
                    mean_j,
                    var_j,
                    spatial_routing_matrix)

    # pose: (N, output_h, output_w, o, 4 x 4) via squeeze mean_j (24, 6, 6, 32, 16)
    poses_j = tf.squeeze(mean_j, axis=-3, name="poses")

    # activation: (N, output_h, output_w, o, 1) via squeeze o_activation is
    # [24, 6, 6, 32, 1]
    activations_j = tf.squeeze(activations_j, axis=-3, name="activations")

    return poses_j, activations_j

  def m_step(self, rr, votes, activations_i, beta_v, beta_a, inverse_temperature):
    """The m-step in EM routing between input capsules (i) and output capsules
    (j).

    Compute the activations of the output capsules (j), and the Gaussians for the
    pose of the output capsules (j).
    See Hinton et al. "Matrix Capsules with EM Routing" for detailed description
    of m-step.

    Author:
      Ashley Gritzman 19/10/2018

    Args:
      rr:
        assignment weights between capsules in layer i and layer j
        (N, output_h, output_w, kernel_h*kernel_w*i, o, 1)
        (64, 6, 6, 9*8, 16, 1)
      votes_ij:
        votes from capsules in layer i to capsules in layer j
        For conv layer:
          (N, output_h, output_w, kernel_h*kernel_w*i, o, 4x4)
          (64, 6, 6, 9*8, 32, 16)
        For FC layer:
          The kernel dimensions are equal to the spatial dimensions of the input
          layer i, and
          the spatial dimensions of the output layer j are 1x1.
          (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
          (64, 1, 1, 4*4*16, 5, 16)
      activations_i:
        activations of capsules in layer i (L)
        (N, output_h, output_w, kernel_h*kernel_w*i, o, n_channels)
        (24, 6, 6, 288, 1, 1)
      beta_v:
        Trainable parameters in computing cost
        (1, 1, 1, 1, 32, 1)
      beta_a:
        Trainable parameters in computing next level activation
        (1, 1, 1, 1, 32, 1)
      inverse_temperature: lambda, increase over each iteration by the caller

    Returns:
      activations_j:
        activations of capsules in layer j (L+1)
        (N, output_h, output_w, 1, o, 1)
        (64, 6, 6, 1, 32, 1)
      mean_j:
        mean of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
      stdv_j:
        standard deviation of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
      var_j:
        variance of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
    """

    with tf.variable_scope("m_step") as scope:
      rr_prime = rr * activations_i
      rr_prime = tf.identity(rr_prime, name="rr_prime")

      # rr_prime_sum: sum over all input capsule i
      rr_prime_sum = tf.reduce_sum(rr_prime,
                                   axis=-3,
                                   keepdims=True,
                                   name='rr_prime_sum')

      # The amount of information given to parent capsules is very different for
      # the final "class-caps" layer. Since all the spatial capsules give output
      # to just a few class caps, they receive a lot more information than the
      # convolutional layers. So in order for lambda and beta_v/beta_a settings to
      # apply to this layer, we must normalise the amount of information.
      # activ from convcaps1 to convcaps2 (64*5*5, 144, 16, 1) 144/16 = 9 info
      # (N*output_h*output_w, kernel_h*kernel_w*i, o, 1)
      # activ from convcaps2 to classcaps (64, 1, 1, 400, 5, 1) 400/5 = 80 info
      # (N, 1, 1, IH*IW*i, n_classes, 1)
      child_caps = float(rr_prime.get_shape().as_list()[-3])
      parent_caps = float(rr_prime.get_shape().as_list()[-2])
      ratio_child_to_parent = child_caps / parent_caps
      layer_norm_factor = 100 / ratio_child_to_parent

      # mean_j: (24, 6, 6, 1, 32, 16)
      mean_j_numerator = tf.reduce_sum(rr_prime * votes,
                                       axis=-3,
                                       keepdims=True,
                                       name="mean_j_numerator")
      mean_j = tf.div(mean_j_numerator,
                      rr_prime_sum + self.hparams.epsilon,
                      name="mean_j")

      # ----- AG 26/06/2018 START -----#
      # Use variance instead of standard deviation, because the sqrt seems to
      # cause NaN gradients during backprop.
      # See original implementation from Suofei below
      var_j_numerator = tf.reduce_sum(rr_prime * tf.math.square(votes - mean_j),
                                      axis=-3,
                                      keepdims=True,
                                      name="var_j_numerator")
      var_j = tf.div(var_j_numerator,
                     rr_prime_sum + self.hparams.epsilon,
                     name="var_j")


      ###################
      # var_j = var_j + 1e-5
      var_j = tf.identity(var_j + 1e-9, name="var_j_epsilon")
      ###################


      ######## layer_norm_factor
      cost_j_h = (beta_v + 0.5 * tf.math.log(var_j)) * rr_prime_sum * layer_norm_factor
      cost_j_h = tf.identity(cost_j_h, name="cost_j_h")

      # ----- END ----- #

      # cost_j: (24, 6, 6, 1, 32, 1)
      # activations_j_cost = (24, 6, 6, 1, 32, 1)
      # yg: This is done for numeric stability.
      # It is the relative variance between each channel determined which one
      # should activate.
      cost_j = tf.reduce_sum(cost_j_h, axis=-1, keepdims=True, name="cost_j")

      activations_j_cost = tf.identity(beta_a - cost_j,
                                       name="activations_j_cost")

      # (24, 6, 6, 1, 32, 1)
      activations_j = tf.sigmoid(inverse_temperature * activations_j_cost,
                                 name="sigmoid")

      return activations_j, mean_j, var_j

  def e_step(self, votes_ij, activations_j, mean_j, stdv_j, var_j, spatial_routing_matrix):
    """The e-step in EM routing between input capsules (i) and output capsules (j).

    Update the assignment weights using in routung. The output capsules (j)
    compete for the input capsules (i).
    See Hinton et al. "Matrix Capsules with EM Routing" for detailed description
    of e-step.

    Author:
      Ashley Gritzman 19/10/2018

    Args:
      votes_ij:
        votes from capsules in layer i to capsules in layer j
        For conv layer:
          (N, output_h, output_w, kernel_h*kernel_w*i, o, 4x4)
          (64, 6, 6, 9*8, 32, 16)
        For FC layer:
          The kernel dimensions are equal to the spatial dimensions of the input
          layer i, and the spatial dimensions of the output layer j are 1x1.
          (N, 1, 1, child_space*child_space*i, output_classes, 4x4)
          (64, 1, 1, 4*4*16, 5, 16)
      activations_j:
        activations of capsules in layer j (L+1)
        (N, output_h, output_w, 1, o, 1)
        (64, 6, 6, 1, 32, 1)
      mean_j:
        mean of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
      stdv_j:
        standard deviation of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
      var_j:
        variance of each channel in capsules of layer j (L+1)
        (N, output_h, output_w, 1, o, n_channels)
        (24, 6, 6, 1, 32, 16)
      spatial_routing_matrix: ???

    Returns:
      rr:
        assignment weights between capsules in layer i and layer j
        (N, output_h, output_w, kernel_h*kernel_w*i, o, 1)
        (64, 6, 6, 9*8, 16, 1)
    """

    with tf.variable_scope("e_step") as scope:
      # AG 26/06/2018: changed stdv_j to var_j
      o_p_unit0 = - tf.reduce_sum(
        tf.square(votes_ij - mean_j, name="num") / (2 * var_j),
        axis=-1,
        keepdims=True,
        name="o_p_unit0")

      o_p_unit2 = - 0.5 * tf.reduce_sum(
        tf.math.log(2 * tf.math.pi * var_j),
        axis=-1,
        keepdims=True,
        name="o_p_unit2"
      )

      # (24, 6, 6, 288, 32, 1)
      o_p = o_p_unit0 + o_p_unit2
      zz = tf.math.log(activations_j + self.hparams.epsilon) + o_p

      # AG 13/11/2018: New implementation of normalising across parents
      # ----- Start -----#
      zz_shape = zz.get_shape().as_list()
      batch_size = zz_shape[0]
      parent_space = zz_shape[1]
      kh_kw_i = zz_shape[3]
      parent_caps = zz_shape[4]
      kk = int(tf.reduce_sum(spatial_routing_matrix[:, 0]))
      child_caps = int(kh_kw_i / kk)

      zz = tf.reshape(zz, [batch_size, parent_space, parent_space, kk,
                           child_caps, parent_caps])


      # In log space
      # Fill the sparse matrix with the smallest value in zz (at least -100)
      sparse_filler = tf.minimum(tf.reduce_min(zz), -100)
      zz_sparse = to_sparse(
        zz,
        spatial_routing_matrix,
        sparse_filler=sparse_filler)

      rr_sparse = softmax_across_parents(zz_sparse, spatial_routing_matrix)

      rr_dense = to_dense(rr_sparse, spatial_routing_matrix)

      rr = tf.reshape(
        rr_dense,
        [batch_size, parent_space, parent_space, kh_kw_i, parent_caps, 1])
      # ----- End -----#

      return rr

