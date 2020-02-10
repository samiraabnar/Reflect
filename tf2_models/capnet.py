import tensorflow as tf
from tensorflow.keras import initializers, layers, models


class Length(layers.Layer):
  """
  Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss
  inputs: shape=[dim_1, ..., dim_{n-1}, dim_n]
  output: shape=[dim_1, ..., dim_{n-1}]
  """

  def call(self, inputs, **kwargs):
    return tf.math.sqrt(tf.reduce_sum(tf.math.square(inputs), -1))

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]


class Mask(layers.Layer):
  """
  Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
  Output shape: [None, d2]
  """

  def call(self, inputs, **kwargs):
    # use true label to select target capsule, shape=[batch_size, num_capsule]
    if type(inputs) is list:  # true label is provided with shape = [batch_size, n_classes], i.e. one-hot code.
      assert len(inputs) == 2
      inputs, mask = inputs
    else:  # if no true label, mask by the max length of vectors of capsules
      x = inputs
      # Enlarge the range of values in x to make max(new_x)=1 and others < 0
      x = (x - tf.reduce_max(x, 1, True)) / tf.epsilon() + 1
      mask = tf.clip(x, 0, 1)  # the max value in x clipped to 1 and other to 0

    # masked inputs, shape = [batch_size, dim_vector]
    inputs_masked = tf.matmul(inputs, mask, [1, 1])
    return inputs_masked

  def compute_output_shape(self, input_shape):
    if type(input_shape[0]) is tuple:  # true label provided
      return tuple([None, input_shape[0][-1]])
    else:
      return tuple([None, input_shape[-1]])


def squash(vectors, axis=-1):
  """
  The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
  :param vectors: some vectors to be squashed, N-dim tensor
  :param axis: the axis to squash
  :return: a Tensor with same shape as input vectors
  """
  s_squared_norm = tf.reduce_sum(tf.math.square(vectors), axis, keepdims=True)
  scale = s_squared_norm / (1 + s_squared_norm) / tf.math.sqrt(s_squared_norm)
  return scale * vectors


class CapsuleLayer(layers.Layer):
  """
  The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
  neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
  from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_vector] and output shape = \
  [None, num_capsule, dim_vector]. For Dense Layer, input_dim_vector = dim_vector = 1.

  :param num_capsule: number of capsules in this layer
  :param dim_vector: dimension of the output vectors of the capsules in this layer
  :param num_routings: number of iterations for the routing algorithm
  """

  def __init__(self, num_capsule, dim_vector, num_routing=3,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               **kwargs):
    super(CapsuleLayer, self).__init__(**kwargs)
    self.num_capsule = num_capsule
    self.dim_vector = dim_vector
    self.num_routing = num_routing
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    print(input_shape)
    assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
    self.input_num_capsule = input_shape[1]
    self.input_dim_vector = input_shape[2]

    # Transform matrix
    print(self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector)
    self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_vector, self.dim_vector],
                             initializer=self.kernel_initializer,
                             name='W')

    # Coupling coefficient. The redundant dimensions are just to facilitate subsequent matrix calculation.
    self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                initializer=self.bias_initializer,
                                name='bias',
                                trainable=False)
    self.built = True

  def call(self, inputs, training=None):
    # inputs.shape=[None, input_num_capsule, input_dim_vector]
    # Expand dims to [None, input_num_capsule, 1, 1, input_dim_vector]
    inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)

    # Replicate num_capsule dimension to prepare being multiplied by W
    # Now it has shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
    inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

    """  
    # Compute `inputs * W` by expanding the first dim of W. More time-consuming and need batch_size.
    # Now W has shape  = [batch_size, input_num_capsule, num_capsule, input_dim_vector, dim_vector]
    w_tiled = tf.tile(tf.expand_dims(self.W, 0), [self.batch_size, 1, 1, 1, 1])

    # Transformed vectors, inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
    inputs_hat = tf.batch_dot(inputs_tiled, w_tiled, [4, 3])
    """
    # Compute `inputs * W` by scanning inputs_tiled on dimension 0. This is faster but requires Tensorflow.
    # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
    inputs_hat = tf.scan(lambda ac, x: tf.matmul(x, self.W, [3, 2]),
                         elems=inputs_tiled,
                         initializer=tf.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))
    """
    # Routing algorithm V1. Use tf.while_loop in a dynamic way.
    def body(i, b, outputs):
        c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
        outputs = squash(tf.reduce_sum(c * inputs_hat, 1, keepdims=True))
        b = b + tf.reduce_sum(inputs_hat * outputs, -1, keepdims=True)
        return [i-1, b, outputs]

    cond = lambda i, b, inputs_hat: i > 0
    loop_vars = [tf.constant(self.num_routing), self.bias, tf.reduce_sum(inputs_hat, 1, keepdims=True)]
    _, _, outputs = tf.while_loop(cond, body, loop_vars)
    """
    # Routing algorithm V2. Use iteration. V2 and V1 both work without much difference on performance
    assert self.num_routing > 0, 'The num_routing should be > 0.'
    for i in range(self.num_routing):
      c = tf.nn.softmax(self.bias, dim=2)  # dim=2 is the num_capsule dimension
      # outputs.shape=[None, 1, num_capsule, 1, dim_vector]
      outputs = squash(tf.reduce_sum(c * inputs_hat, 1, keepdims=True))

      # last iteration needs not compute bias which will not be passed to the graph any more anyway.
      if i != self.num_routing - 1:
        # self.bias = tf.update_add(self.bias, tf.reduce_sum(inputs_hat * outputs, [0, -1], keepdims=True))
        self.bias += tf.reduce_sum(inputs_hat * outputs, -1, keepdims=True)
      # tf.reduce_summary.histogram('BigBee', self.bias)  # for debugging
    return tf.reshape(outputs, [-1, self.num_capsule, self.dim_vector])

  def compute_output_shape(self, input_shape):
    return tuple([None, self.num_capsule, self.dim_vector])


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
  """
  Apply Conv2D `n_channels` times and concatenate all capsules
  :param inputs: 4D tensor, shape=[None, width, height, channels]
  :param dim_vector: the dim of the output vector of capsule
  :param n_channels: the number of types of capsules
  :return: output tensor, shape=[None, num_capsule, dim_vector]
  """
  print(inputs, dim_vector, n_channels, kernel_size, strides, padding)
  output = layers.Conv2D(filters=dim_vector * n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(
    inputs)
  print(output)
  ts = output.shape[1]*output.shape[2]*output.shape[3] // dim_vector
  outputs = layers.Reshape(target_shape=[ts, dim_vector])(output)
  print(outputs)
  return layers.Lambda(squash)(outputs)

def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)
    print(x)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    print(conv1)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    print(primarycaps)
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[28, 28, 1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])