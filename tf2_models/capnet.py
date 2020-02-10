import tensorflow as tf
from tensorflow.keras import initializers, layers, models
import tensorflow.keras.backend as K


class Length(layers.Layer):
  """
  Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
  Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
  inputs: shape=[None, num_vectors, dim_vector]
  output: shape=[None, num_vectors]
  """

  def call(self, inputs, **kwargs):
    return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def get_config(self):
    config = super(Length, self).get_config()
    return config


class Mask(layers.Layer):
  """
  Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
  input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
  masked Tensor.
  For example:
      ```
      x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
      y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
      out = Mask()(x)  # out.shape=[8, 6]
      # or
      out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
      ```
  """

  def call(self, inputs, **kwargs):
    if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
      assert len(inputs) == 2
      inputs, mask = inputs
    else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
      # compute lengths of capsules
      x = K.sqrt(K.sum(K.square(inputs), -1))
      # generate the mask which is a one-hot code.
      # mask.shape=[None, n_classes]=[None, num_capsule]
      mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

    # inputs.shape=[None, num_capsule, dim_capsule]
    # mask.shape=[None, num_capsule]
    # masked.shape=[None, num_capsule * dim_capsule]
    masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
    return masked

  def compute_output_shape(self, input_shape):
    if type(input_shape[0]) is tuple:  # true label provided
      return tuple([None, input_shape[0][1] * input_shape[0][2]])
    else:  # no true label provided
      return tuple([None, input_shape[1] * input_shape[2]])

  def get_config(self):
    config = super(Mask, self).get_config()
    return config

def squash(vectors, axis=-1):
  """
  The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
  :param vectors: some vectors to be squashed, N-dim tensor
  :param axis: the axis to squash
  :return: a Tensor with same shape as input vectors
  """
  s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
  scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
  return scale * vectors

class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.

    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
      super(CapsuleLayer, self).__init__(**kwargs)
      self.num_capsule = num_capsule
      self.dim_capsule = dim_capsule
      self.routings = routings
      self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
      print(input_shape)
      assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
      self.input_num_capsule = input_shape[1]
      self.input_dim_capsule = input_shape[2]

      # Transform matrix
      self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                      self.dim_capsule, self.input_dim_capsule],
                               initializer=self.kernel_initializer,
                               name='W')

      self.built = True

    def call(self, inputs, training=None):
      # inputs.shape=[None, input_num_capsule, input_dim_capsule]
      # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
      inputs_expand = K.expand_dims(inputs, 1)

      # Replicate num_capsule dimension to prepare being multiplied by W
      # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
      inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])

      # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
      # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
      # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
      # Regard the first two dimensions as `batch` dimension,
      # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
      # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
      inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

      # Begin: Routing algorithm ---------------------------------------------------------------------#
      # The prior for coupling coefficient, initialized as zeros.
      # b.shape = [None, self.num_capsule, self.input_num_capsule].
      b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

      assert self.routings > 0, 'The routings should be > 0.'
      for i in range(self.routings):
        # c.shape=[batch_size, num_capsule, input_num_capsule]
        print(c)
        c = tf.nn.softmax(b, axis=1)

        # c.shape =  [batch_size, num_capsule, input_num_capsule]
        # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
        # The first two dimensions as `batch` dimension,
        # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
        # outputs.shape=[None, num_capsule, dim_capsule]
        outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]
        print(outputs)
        print(b)
        if i < self.routings - 1:
          # outputs.shape =  [None, num_capsule, dim_capsule]
          # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
          # The first two dimensions as `batch` dimension,
          # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
          # b.shape=[batch_size, num_capsule, input_num_capsule]
          b += K.batch_dot(outputs, inputs_hat, [2, 3])
      # End: Routing algorithm -----------------------------------------------------------------------#

      return outputs

    def compute_output_shape(self, input_shape):
      return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
      config = {
        'num_capsule': self.num_capsule,
        'dim_capsule': self.dim_capsule,
        'routings': self.routings
      }
      base_config = super(CapsuleLayer, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
  """
  Apply Conv2D `n_channels` times and concatenate all capsules
  :param inputs: 4D tensor, shape=[None, width, height, channels]
  :param dim_capsule: the dim of the output vector of capsule
  :param n_channels: the number of types of capsules
  :return: output tensor, shape=[None, num_capsule, dim_capsule]
  """
  output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                         name='primarycap_conv2d')(inputs)
  ts = output.shape[1]*output.shape[2]*output.shape[3] // dim_capsule
  outputs = layers.Reshape(target_shape=[ts, dim_capsule], name='primarycap_reshape')(output)
  return layers.Lambda(squash, name='primarycap_squash')(outputs)

def CapsNet(input_shape, n_class, num_routing):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 4d, [None, width, height, channels]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    # Layer 3: Capsule layer. Routing algorithm works here.
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=num_routing,
                             name='digitcaps')(primarycaps)
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