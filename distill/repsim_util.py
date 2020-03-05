import tensorflow as tf

def get_reps(inputs, model, index=1, layer=None):
  """
  If Model is LSTM:
      1: final_rnn_outputs,
      2: hidden_activation (for all layers, including input embeddings)
  """
  outputs = model.detailed_call(inputs)

  if index is not tuple:
    index = (index,)
    layer = (layer,)

  reps = ()
  for i,l in zip(index,layer):
    rep = outputs[i]

    if l is not None:
      rep = rep[l]
      
    reps = reps + (rep,)

  return reps


def normalized_pairwisedot_product_sim(reps1, reps2):
  reps1 = reps1 / tf.norm(reps1, axis=-1)[..., None]
  reps2 = reps2 / tf.norm(reps2, axis=-1)[..., None]

  pw_dot_product = tf.cast(tf.matmul(reps1, reps2, transpose_b=True), dtype=tf.float32)

  return pw_dot_product


def normalized_dot_product_sim(reps1, reps2, padding_mask):
  # normalize reps:
  reps1 = reps1 / tf.norm(reps1, axis=-1)[..., None]
  reps2 = reps2 / tf.norm(reps2, axis=-1)[..., None]

  # Elementwise multiplication
  dot_product = tf.multiply(reps1, reps2)

  # Sum over last axis to get the dot product similarity between corresponding pairs
  dot_product = tf.reduce_sum(dot_product, axis=-1)
  dot_product = tf.multiply(dot_product, padding_mask[:, 0])

  return dot_product


def second_order_rep_sim(reps1, reps2, padding_mask):
  sims1 = normalized_pairwisedot_product_sim(reps1, reps1)
  sims2 = normalized_pairwisedot_product_sim(reps2, reps2)

  padding_mask = tf.ones((tf.shape(reps1)[0], 1))
  so_sims = normalized_dot_product_sim(sims1, sims2, padding_mask) * padding_mask[:, 0]
  mean_sim = tf.reduce_sum(so_sims) / tf.reduce_sum(padding_mask)

  return mean_sim, so_sims


def compare_models(inputs, model1, model2, index1=1, index2=1, layer1=None, layer2=None, padding_symbol=None):
  reps1 = get_reps(inputs, model1, index=index1, layer=layer1)
  reps2 = get_reps(inputs, model2, index=index2, layer=layer2)

  reps1 = tf.reshape(reps1, (-1, tf.shape(reps1)[-1]))
  reps2 = tf.reshape(reps2, (-1, tf.shape(reps2)[-1]))

  if padding_symbol is not None and padding_symbol > -1:
    padding_mask = tf.cast(1.0 - (inputs == padding_symbol), dtype=tf.float32)
    padding_mask = tf.reshape(padding_mask, (-1, 1))
  else:
    padding_mask = tf.ones((tf.shape(reps1)[0]))

  similarity_measures = second_order_rep_sim(reps1, reps2, padding_mask=padding_mask)

  return similarity_measures


def compare_reps(reps1, reps2, padding_symbol=None, inputs=None):
  reps1 = tf.reshape(reps1, (-1, tf.shape(reps1)[-1]))
  reps2 = tf.reshape(reps2, (-1, tf.shape(reps2)[-1]))

  if padding_symbol is not None and padding_symbol > -1:
    padding_mask = tf.cast(1.0 - (inputs == padding_symbol), dtype=tf.float32)
    padding_mask = tf.reshape(padding_mask, (-1, 1))
  else:
    padding_mask = tf.ones((tf.shape(reps1)[0], 1))

  similarity_measures = second_order_rep_sim(reps1, reps2, padding_mask)

  return similarity_measures

@tf.function
def rep_loss(reps1, reps2, padding_symbol=None, inputs=None):
  reps1 = tf.reshape(reps1, (-1, tf.shape(reps1)[-1]))
  reps2 = tf.reshape(reps2, (-1, tf.shape(reps2)[-1]))

  if padding_symbol is not None and padding_symbol > -1:
    padding_mask = tf.cast(1.0 - (inputs == padding_symbol), dtype=tf.float32)
    padding_mask = tf.reshape(padding_mask, (-1, 1))
  else:
    padding_mask = tf.ones((tf.shape(reps1)[0], 1))
  mean_sim, _ = second_order_rep_sim(reps1, reps2, padding_mask)

  return 1.0 - mean_sim