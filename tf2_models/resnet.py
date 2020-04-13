import tensorflow as tf

class ResnetBlock(tf.keras.layers.Layer):
  def __init__(self, filters, conv_size, activation='relu',*inputs, **kwargs):
    super(ResnetBlock, self).__init__(*inputs, **kwargs)
    self.filter = filters
    self.conv_size = conv_size
    self.activation = activation
    self.create_layer()

  def create_layer(self):
    self.conv1 = tf.keras.layers.Conv2D(self.filters, self.conv_size, activation=self.activation, padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization()
    self.conv2 = tf.keras.Conv2D(self.filters, self.conv_size, activation=None, padding='same')

  def call(self, inputs, training=None, **kwargs):
    outputs = self.conv1(inputs, training=None, **kwargs)
    outputs = self.batch_norm(outputs,training=None, **kwargs)
    outputs = tf.keras.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    outputs = tf.keras.BatchNormalization()(x)
    outputs = tf.keras.Add()([x, input_data])
    outputs = tf.keras.Activation('relu')(x)

    return outputs
