import tensorflow as tf
import numpy as np

def create_routing_map(child_space, k, s):
  """Generate TFRecord for train and test datasets from .mat files.

  Create a binary map where the rows are capsules in the lower layer (children)
  and the columns are capsules in the higher layer (parents). The binary map
  shows which children capsules are connected to which parent capsules along the   spatial dimension.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    child_space: spatial dimension of lower capsule layer
    k: kernel size
    s: stride
  Returns:
    binmap:
      A 2D numpy matrix containing mapping between children capsules along the
      rows, and parent capsules along the columns.
      (child_space^2, parent_space^2)
      (7*7, 5*5)
  """

  parent_space = tf.cast((child_space - k) / s + 1, tf.int32)
  #binmap = np.zeros((child_space ** 2, parent_space ** 2))
  binmap = tf.TensorArray(size=0, dynamic_size=True ,dtype=tf.float32, infer_shape=True)
  c_eye = tf.eye(child_space ** 2)
  for r in range(parent_space):
    for c in range(parent_space):
      p_idx = r * parent_space + c
      binmap = binmap.write(p_idx, tf.zeros(child_space ** 2))
      for i in range(k):
        # c_idx stand for child_index; p_idx is parent_index
        c_idx = r * s * child_space + c * s + child_space * i
        all_c_idx = tf.range((c_idx),(c_idx + k))
        all_c = tf.gather(c_eye, all_c_idx)
        new_row = tf.reduce_sum(all_c, axis=0)
        prev = binmap.read(p_idx)
        binmap = binmap.write(p_idx,new_row + prev)

  binmap = binmap.stack()
  tf.print('binmap',tf.shape(binmap), child_space ** 2, parent_space ** 2)
  binmap = tf.transpose(binmap)
  tf.print('binmap',tf.shape(binmap), child_space ** 2, parent_space ** 2, tf.reduce_sum(binmap, axis=0))

  return binmap

def kernel_tile(input, kernel, stride):
  """Tile the children poses/activations so that the children for each parent occur in one axis.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    input:
      tensor of child poses or activations
      poses (N, child_space, child_space, i, 4, 4) -> (64, 7, 7, 8, 4, 4)
      activations (N, child_space, child_space, i, 1) -> (64, 7, 7, 8, 16)
    kernel:
    stride:
  Returns:
    tiled:
      (N, parent_space, parent_space, kh*kw, i, 16 or 1)
      (64, 5, 5, 9, 8, 16 or 1)
    child_parent_matrix:
      A 2D numpy matrix containing mapping between children capsules along the
      rows, and parent capsules along the columns.
      (child_space^2, parent_space^2)
      (7*7, 5*5)
  """

  input_shape = tf.shape(input) #2 14 14 8 16
  batch_size = input_shape[0] #2
  spatial_size = input_shape[1] #14
  n_capsules = input_shape[3] # 8
  tf.print('kernel tile', batch_size, spatial_size, n_capsules)
  # parent_spatial_size ((14 - 3) / 2)+1 --> 6
  parent_spatial_size = tf.cast((spatial_size - kernel) / stride + 1, dtype=tf.int32)

  # Check that dim 1 and 2 correspond to the spatial size
  #assert input_shape[1] == input_shape[2]

  # # Check if we have poses or activations
  # if len(input_shape) > 5:
  #   # Poses
  #   size = input_shape[4] * input_shape[5]
  # else:
  #   # Activations
  #   size = 1

  # Matrix showing which children map to which parent. Children are rows,
  # parents are columns.
  # 196 x 36
  child_parent_matrix = create_routing_map(spatial_size, kernel, stride)
  tf.print('child_parent_matrix', tf.shape(child_parent_matrix))
  # Convert from np to tf
  # child_parent_matrix = tf.constant(child_parent_matrix)

  # Each row contains the children belonging to one parent
  # 36 x 6
  child_to_parent_idx = group_children_by_parent(child_parent_matrix)
  tf.print('child_to_parent_idx', tf.shape(child_to_parent_idx)) #36 6

  tf.print('input', tf.shape(input), spatial_size, spatial_size * spatial_size)
  # Spread out spatial dimension of children
  input = tf.reshape(input, [batch_size, spatial_size * spatial_size, -1])
  tf.print('input', tf.shape(input)) #2 14*14 128

  # Select which children go to each parent capsule
  tf.print('input', tf.shape(input))
  tiled = tf.gather(input, child_to_parent_idx, axis=1) #2 196 128

  tf.print('tiled', tf.shape(tiled))
  tiled = tf.squeeze(tiled)
  tf.print('tiled', tf.shape(tiled))
  tiled = tf.reshape(tiled, [batch_size, parent_spatial_size, parent_spatial_size, kernel * kernel, n_capsules, -1])
  tf.print('tiled', tf.shape(tiled))

  return tiled, child_parent_matrix

def group_children_by_parent(bin_routing_map):
  """Groups children capsules by parent capsule.

  Rearrange the bin_routing_map so that each row represents one parent capsule,
  and the entries in the row are indexes of the children capsules that route to
  that parent capsule. This mapping is only along the spatial dimension, each
  child capsule along in spatial dimension will actually contain many capsules,
  e.g. 32. The grouping that we are doing here tell us about the spatial
  routing, e.g. if the lower layer is 7x7 in spatial dimension, with a kernel of
  3 and stride of 1, then the higher layer will be 5x5 in the spatial dimension.
  So this function will tell us which children from the 7x7=49 lower capsules
  map to each of the 5x5=25 higher capsules. One child capsule can be in several
  different parent capsules, children in the corners will only belong to one
  parent, but children towards the center will belong to several with a maximum
  of kernel*kernel (e.g. 9), but also depending on the stride.

  Author:
    Ashley Gritzman 19/10/2018
  Args:
    bin_routing_map:
      binary routing map with children as rows and parents as columns
  Returns:
    children_per_parents:
      parents are rows, and the indexes in the row are which children belong to that parent
  """

  true_indexes = tf.where(tf.transpose(bin_routing_map))
  children_per_parent = tf.reshape(true_indexes, [tf.shape(bin_routing_map)[1], -1])
  tf.print('true_indexes', tf.shape(true_indexes))
  return children_per_parent

def to_sparse(probs, spatial_routing_matrix, sparse_filler=tf.math.log(1e-20)):
  """Convert probs tensor to sparse along child_space dimension.

  Consider a probs tensor of shape (64, 6, 6, 3*3, 32, 16).
  (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
  parent_caps)
  The tensor contains the probability of each child capsule belonging to a
  particular parent capsule. We want to be able to sum the total probabilities
  for a single child capsule to all the parent capsules. So we need to convert
  the 3*3 spatial locations have been condensed, into a sparse format across
  all child spatial location e.g. 14*14.

  Since we are working in log space, we must replace the zeros that come about
  during sparse with log(0). The 'sparse_filler' option allows us to specify the
  number to use to fill.

  Author:
    Ashley Gritzman 01/11/2018

  Args:
    probs:
      tensor of log probabilities of each child capsule belonging to a
      particular parent capsule
      (batch_size, parent_space, parent_space, kernel*kernel, child_caps,
      parent_caps)
      (64, 5, 5, 3*3, 32, 16)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns
    sparse_filler:
      the number to use to fill in the sparse locations instead of zero

  Returns:
    sparse:
      the sparse representation of the probs tensor in log space
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 7*7, 32, 16)
  """

  # Get shapes of probs
  shape = tf.shape(probs)
  batch_size = shape[0]
  parent_space = shape[1]
  kk = shape[3]
  child_caps = shape[4]
  parent_caps = shape[5]

  # Get spatial dimesion of child capsules
  srm_shape = tf.shape(spatial_routing_matrix)
  child_space_2 = srm_shape[0]
  parent_space_2 =  srm_shape[1]

  # Unroll the probs along the spatial dimension
  # e.g. (64, 6, 6, 3*3, 8, 32) -> (64, 6*6, 3*3, 8, 32)
  probs_unroll = tf.reshape(
    probs,
    [batch_size, parent_space_2, kk, child_caps, parent_caps])

  # Each row contains the children belonging to one parent
  child_to_parent_idx = group_children_by_parent(spatial_routing_matrix)

  # Create an index mapping each capsule to the correct sparse location
  # Each element of the index must contain [batch_position,
  # parent_space_position, child_sparse_position]
  # E.g. [63, 24, 49] maps image 63, parent space 24, sparse position 49
  child_sparse_idx = child_to_parent_idx
  child_sparse_idx = child_sparse_idx[tf.newaxis, ...]
  child_sparse_idx = tf.tile(child_sparse_idx, [batch_size, 1, 1])

  parent_idx = tf.range(parent_space_2)
  parent_idx = tf.reshape(parent_idx, [-1, 1])
  parent_idx = tf.repeat(parent_idx, kk)
  parent_idx = tf.tile(parent_idx, [batch_size])

  tf.print('parent_idx', parent_idx.shape)
  parent_idx = tf.reshape(parent_idx, [batch_size, parent_space_2, kk])
  tf.print('parent_idx', parent_idx.shape)


  batch_idx = tf.range(batch_size)
  tf.print('batch_idx', batch_idx.shape)
  batch_idx = tf.reshape(batch_idx, [-1, 1])
  batch_idx = tf.tile(batch_idx, [1,parent_space_2 * kk])
  batch_idx = tf.reshape(batch_idx, [batch_size, parent_space_2, kk])
  tf.print('batch_idx', batch_idx.shape)

  # Combine the 3 coordinates
  indices = tf.stack((batch_idx, parent_idx, tf.cast(child_sparse_idx, dtype=tf.int32)), axis=3)
  #indices = tf.constant(indices)

  # Convert each spatial location to sparse
  shape = [batch_size, parent_space_2, child_space_2, child_caps, parent_caps]
  sparse = tf.scatter_nd(indices, probs_unroll, shape)

  # scatter_nd pads the output with zeros, but since we are operating
  # in log space, we need to replace 0 with log(0), or log(1e-9)
  zeros_in_log = tf.ones_like(sparse, dtype=tf.float32) * sparse_filler
  sparse = tf.where(tf.equal(sparse, 0.0), zeros_in_log, sparse)

  # Reshape
  # (64, 5*5, 7*7, 8, 32) -> (64, 6, 6, 14*14, 8, 32)
  sparse = tf.reshape(sparse, [batch_size, parent_space, parent_space, child_space_2, child_caps, parent_caps])

  # Checks
  # 1. Shape
  # assert [a for a in tf.shape(sparse)] == [batch_size, parent_space, parent_space, child_space_2, child_caps,
  #                                         parent_caps]

  tf.print("end of sparse")
  return sparse


def normalise_across_parents(probs_sparse, spatial_routing_matrix):
  """Normalise across all parent capsules including spatial and depth.

  Consider a sparse matrix of probabilities (1, 5, 5, 49, 8, 32)
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,   parent_caps)
  For one child capsule, we need to normalise across all parent capsules that
  receive output from that child. This includes the depth of parent capsules,
  and the spacial dimension od parent capsules. In the example matrix of
  probabilities above this would mean normalising across [1, 2, 5] or
  [parent_space, parent_space, parent_caps].

  Author:
    Ashley Gritzman 05/11/2018
  Args:
    probs_sparse:
      the sparse representation of the probs matrix, not in log
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)

  Returns:
    rr_updated:
      softmax across all parent capsules, same shape as input
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
  """

  # e.g. (1, 5, 5, 49, 8, 32)
  # (batch_size, parent_space, parent_space, child_space*child_space, child_caps, parent_caps)
  shape = tf.shape(probs_sparse)
  batch_size = shape[0]
  parent_space = shape[1]
  child_space_2 = shape[3]  # squared
  child_caps = shape[4]
  parent_caps = shape[5]

  rr_updated = probs_sparse / (tf.reduce_sum(probs_sparse,
                                             axis=[1, 2, 5],
                                             keepdims=True) + 1e-9)

  # Checks
  # 1. Shape
  # assert ([a for a in tf.shape(rr_updated)]
  #         == [batch_size, parent_space, parent_space, child_space_2,
  #             child_caps, parent_caps])

  # 2. Total of routing weights must equal number of child capsules minus
  # dropped ones.
  # Because of numerical issues it is not likely that the routing weights will
  # equal the calculated number of capsules, so we check that it is within a
  # certain percent.
  dropped_child_caps = tf.reduce_sum(tf.reduce_sum(spatial_routing_matrix, axis=1) < 1e-9)
  # effective_child_caps = (child_space_2 - dropped_child_caps) * child_caps *
  # batch_size
  effective_child_caps = (child_space_2 - dropped_child_caps) * child_caps
  effective_child_caps = tf.to_double(effective_child_caps)

  sum_routing_weights = tf.reduce_sum(tf.to_double(rr_updated),
                                      axis=[1, 2, 3, 4, 5])

  pct_delta = tf.abs((effective_child_caps - sum_routing_weights)
                     / effective_child_caps)

  #   assert_op = tf.assert_less(
  #       pct_delta,
  #       tf.to_double(0.01),
  #       message="""function normalise_across_parents: total of routing weights
  #               not equal to number of child capsules""",
  #       data=[pct_delta, sum_routing_weights, effective_child_caps,
  #             tf.reduce_min(sum_routing_weights)],
  #       summarize=10)
  #   with tf.control_dependencies([assert_op]):
  #       rr_updated = tf.identity(rr_updated)

  return rr_updated


def softmax_across_parents(probs_sparse, spatial_routing_matrix):
  """Softmax across all parent capsules including spatial and depth.

  Consider a sparse matrix of probabilities (1, 5, 5, 49, 8, 32)
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,   parent_caps)
  For one child capsule, we need to normalise across all parent capsules that
  receive output from that child. This includes the depth of parent capsules,
  and the spacial dimension od parent capsules. In the example matrix of
  probabilities above this would mean normalising across [1, 2, 5] or
  [parent_space, parent_space, parent_caps]. But the softmax function
  `tf.nn.softmax` can only operate across one axis, so we need to reshape the
  matrix such that we can combine paret_space and parent_caps into one axis.

  Author:
    Ashley Gritzman 05/11/2018

  Args:
    probs_sparse:
      the sparse representation of the probs matrix, in log
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)

  Returns:
    rr_updated:
      softmax across all parent capsules, same shape as input
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
  """

  # e.g. (1, 5, 5, 49, 8, 32)
  # (batch_size, parent_space, parent_space, child_space*child_space,
  # child_caps, parent_caps)
  tf.print('softmax across parents')
  shape = tf.shape(probs_sparse)
  batch_size = shape[0]
  parent_space = shape[1]
  child_space_2 = shape[3]  # squared
  child_caps = shape[4]
  parent_caps = shape[5]

  # Move parent space dimensions, and parent depth dimension to end
  # (1, 5, 5, 49, 8, 32)  -> (1, 49, 4, 5, 5, 3)
  sparse = tf.transpose(probs_sparse, perm=[0, 3, 4, 1, 2, 5])

  # Combine parent
  # (1, 49, 4, 75)
  tf.print('sparse', sparse.shape)
  sparse = tf.reshape(sparse, [batch_size, child_space_2, child_caps, -1])
  tf.print('sparse', sparse.shape)

  # Perform softmax across parent capsule dimension
  parent_softmax = tf.nn.softmax(sparse, axis=-1)

  # Uncombine parent space and depth
  # (1, 49, 4, 5, 5, 3)
  tf.print('parent_softmax', parent_softmax.shape)
  parent_softmax = tf.reshape(
    parent_softmax,
    [batch_size, child_space_2, child_caps, parent_space, parent_space,
     parent_caps])
  tf.print('parent_softmax', parent_softmax.shape)

  # Return to original order
  # (1, 5, 5, 49, 8, 32)
  parent_softmax = tf.transpose(parent_softmax, perm=[0, 3, 4, 1, 2, 5])

  # Softmax across the parent capsules actually gives us the updated routing
  # weights
  rr_updated = parent_softmax

  # Checks
  # 1. Shape
  # assert ([a for a in tf.shape(rr_updated)]
  #         == [batch_size, parent_space, parent_space, child_space_2,
  #             child_caps, parent_caps])

  # 2. Check the total of the routing weights is equal to the number of child
  # capsules
  # Note: during convolution some child capsules may be dropped if the
  # convolution doesn't fit nicely. So in the sparse form of child capsules, the   # dropped capsules will be 0 everywhere. When we do a softmax, these capsules
  # will then be given a value, so when we check the total child capsules we
  # need to include these. But these will then be excluded when we convert back   # to dense so it's not a problem.
  #total_child_caps = tf.cast(child_space_2 * child_caps * batch_size, dtype=tf.float32)
  #sum_routing_weights = tf.round(tf.reduce_sum(rr_updated))

  #   assert_op = tf.assert_equal(
  #       sum_routing_weights,
  #       total_child_caps,
  #       message="""in fn softmax_across_parents: sum_routing_weights and
  #               effective_child_caps are different""")
  #   with tf.control_dependencies([assert_op]):
  #      rr_updated = tf.identity(rr_updated)

  tf.print('end of softmax across parents')
  return rr_updated


def to_dense(sparse, spatial_routing_matrix):
  """Convert sparse back to dense along child_space dimension.

  Consider a sparse probs tensor of shape (64, 5, 5, 49, 8, 32).
  (batch_size, parent_space, parent_space, child_space*child_space, child_caps,
  parent_caps)
  The tensor contains all child capsules at every parent spatial location, but
  if the child does not route to the parent then it is just zero at that spot.
  Now we want to get back to the dense representation:
  (64, 5, 5, 49, 8, 32) -> (64, 5, 5, 9, 8, 32)

  Author:
    Ashley Gritzman 05/11/2018
  Args:
    sparse:
      the sparse representation of the probs tensor
      (batch_size, parent_space, parent_space, child_space*child_space,
      child_caps, parent_caps)
      (64, 5, 5, 49, 8, 32)
    spatial_routing_matrix:
      binary routing map with children as rows and parents as columns

  Returns:
    dense:
      the dense representation of the probs tensor
      (batch_size, parent_space, parent_space, kk, child_caps, parent_caps)
      (64, 5, 5, 9, 8, 32)
  """

  # Get shapes of probs
  shape = tf.shape(sparse)
  batch_size = shape[0]
  parent_space = shape[1]
  child_space_2 = shape[3]  # squared
  child_caps = shape[4]
  parent_caps = shape[5]

  # Calculate kernel size by adding up column of spatial routing matrix
  kk = tf.cast(tf.reduce_sum(spatial_routing_matrix[:, 0]), dtype=tf.int32)

  # Unroll parent spatial dimensions
  # (64, 5, 5, 49, 8, 32) -> (64, 5*5, 49, 8, 32)
  sparse_unroll = tf.reshape(sparse, [batch_size, parent_space * parent_space,
                                      child_space_2, child_caps, parent_caps])

  # Apply boolean_mask on axis 1 and 2
  # sparse_unroll: (64, 5*5, 49, 8, 32)
  # spatial_routing_matrix: (49, 25) -> (25, 49)
  # dense: (64, 5*5, 49, 8, 32) -> (64, 5*5*9, 8, 32)
  dense = tf.boolean_mask(sparse_unroll,
                          tf.transpose(spatial_routing_matrix), axis=1)

  # Reshape
  dense = tf.reshape(dense, [batch_size, parent_space, parent_space, kk,
                             child_caps, parent_caps])

  # Checks
  # 1. Shape
  # assert ([a for a in tf.shape(dense)]
  #         == [batch_size, parent_space, parent_space, kk, child_caps,
  #             parent_caps])

  #   # 2. Total of dense and sparse must be the same
  #   delta = tf.abs(tf.reduce_sum(dense, axis=[3])
  #                  - tf.reduce_sum(sparse, axis=[3]))
  #   assert_op = tf.assert_less(
  #       delta,
  #       1e-6,
  #       message="in fn to_dense: total of dense and sparse are different",
  #       data=[tf.reduce_sum(dense,[1,2,3,4,5]),
  #             tf.reduce_sum(sparse,[1,2,3,4,5]),
  #             tf.reduce_sum(dense),tf.reduce_sum(sparse)],
  #       summarize=10)
  #   with tf.control_dependencies([assert_op]):
  #      dense = tf.identity(dense)

  return dense


def logits_one_vs_rest(logits, positive_class=0):
  """Return the logit from the positive class and the maximum logit from the
  other classes.

  This function is used to prepare the logits from a multi class classifier to
  be used for binary classification. The logits from the positive class are
  placed in column 0. The maximum logit from the remaining classes is placed in   column 1.

  Author:
    Ashley Gritzman 04/12/2018
  Args:
    logits_all: logits from multiple classes
    positive_class: the index of the positive class
  Returns:
    logits_one_vs_rest:
      logits from positive class in column 0, and maximum logits of other
      classes in column 1
  """

  logits_positive = tf.reshape(logits[:, positive_class], [-1, 1])

  logits_rest = tf.concat([logits[:, :positive_class],
                           logits[:, (positive_class + 1):]], axis=1)
  logits_rest_max = tf.reduce_max(logits_rest, axis=1, keepdims=True)

  logits_one_vs_rest = tf.concat([logits_positive, logits_rest_max], axis=1)

  return logits_one_vs_rest

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

  srm_shape = tf.shape(spatial_routing_matrix)
  # Get spatial dimension of parent & child
  parent_space_2 = tf.cast(srm_shape[1], tf.int32)
  parent_space = tf.cast(tf.math.sqrt(parent_space_2), tf.int32)
  child_space_2 = tf.cast(srm_shape[0], tf.int32)
  child_space = tf.cast(tf.math.sqrt(child_space_2), tf.int32)

  # Count the number of parents that each child belongs to
  parents_per_child = tf.reduce_sum(spatial_routing_matrix, axis=1, keepdims=True)

  # Divide the vote of each child by the number of parents that it belongs to
  # If the striding causes the filter not to fit, it will result in some
  # "dropped" child capsules, which effectively means child capsules that do not
  # have any parents. This would create a divide by 0 scenario, so need to add
  # 1e-9 to prevent NaNs.
  rr_initial = (spatial_routing_matrix
                / (parents_per_child * parent_caps + 1e-9))

  # Convert the sparse matrix to be compatible with votes.
  # This is done by selecting the child capsules belonging to each parent, which
  # is achieved by selecting the non-zero values down each column. Need the
  # combination of two transposes so that order is correct when reshaping
  mask = spatial_routing_matrix.astype(bool)
  rr_initial = rr_initial.T[mask.T]
  rr_initial = tf.reshape(rr_initial, [parent_space, parent_space, -1])

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
  dropped_child_caps = tf.reduce_sum(tf.reduce_sum(spatial_routing_matrix, axis=1) < 1e-9)
  effective_child_cap = ((child_space * child_space - dropped_child_caps)
                         * child_caps)

  sum_routing_weights = tf.reduce_sum(rr_initial)

  #   assert_op = tf.assert_less(
  #       np.abs(sum_routing_weights - effective_child_cap), 1e-9)
  #   with tf.control_dependencies([assert_op]):
  #     return rr_initial

  # assert tf.abs(sum_routing_weights - effective_child_cap) < 1e-3

  return rr_initial


# ------------------------------------------------------------------------------
# LOSS FUNCTIONS
# ------------------------------------------------------------------------------
def spread_loss(scores, y):
  """Spread loss.

  "In order to make the training less sensitive to the initialization and
  hyper-parameters of the model, we use “spread loss” to directly maximize the
  gap between the activation of the target class (a_t) and the activation of the
  other classes. If the activation of a wrong class, a_i, is closer than the
  margin, m, to at then it is penalized by the squared distance to the margin."

  See Hinton et al. "Matrix Capsules with EM Routing" equation (3).

  Author:
    Ashley Gritzman 19/10/2018
  Credit:
    Adapted from Suofei Zhang's implementation on GitHub, "Matrix-Capsules-
    EM-Tensorflow"
    https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
  Args:
    scores:
      scores for each class
      (batch_size, num_class)
    y:
      index of true class
      (batch_size, 1)
  Returns:
    loss:
      mean loss for entire batch
      (scalar)
  """

  batch_size = tf.shape(scores)[0]

  # AG 17/09/2018: modified margin schedule based on response of authors to
  # questions on OpenReview.net:
  # https://openreview.net/forum?id=HJWLfGWRb
  # "The margin that we set is:
  # margin = 0.2 + .79 * tf.sigmoid(tf.minimum(10.0, step / 50000.0 - 4))
  # where step is the training step. We trained with batch size of 64."
  global_step = tf.to_float(tf.train.get_global_step())
  m_min = 0.2
  m_delta = 0.79
  m = (m_min
       + m_delta * tf.sigmoid(tf.minimum(10.0, global_step / 50000.0 - 4)))

  num_class = int(tf.shape(scores)[-1])

  y = tf.one_hot(y, num_class, dtype=tf.float32)

  # Get the score of the target class
  # (64, 1, 5)
  scores = tf.reshape(scores, shape=[batch_size, 1, num_class])
  # (64, 5, 1)
  y = tf.expand_dims(y, axis=2)
  # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
  at = tf.matmul(scores, y)

  # Compute spread loss, paper eq (3)
  loss = tf.square(tf.maximum(0., m - (at - scores)))

  # Sum losses for all classes
  # (64, 1, 5)*(64, 5, 1) = (64, 1, 1)
  # e.g loss*[1 0 1 1 1]
  loss = tf.matmul(loss, 1. - y)

  # Compute mean
  loss = tf.reduce_mean(loss)

  return loss


def cross_ent_loss(logits, y):
  """Cross entropy loss.

  Author:
    Ashley Gritzman 06/05/2019
  Args:
    logits:
      logits for each class
      (batch_size, num_class)
    y:
      index of true class
      (batch_size, 1)
  Returns:
    loss:
      mean loss for entire batch
      (scalar)
  """
  loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
  loss = tf.reduce_mean(loss)

  return loss



