import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class RandomGaussian(tfds.core.GeneratorBasedBuilder):
  """Short description of my dataset."""

  VERSION = tfds.core.Version('0.1.0')
  INPUT_DIM=64
  NUMBER_OF_EXAMPLES = 100000
  SEQ_LEN = 20
  NUM_CLASSES=3

  def _info(self):
    return tfds.core.DatasetInfo(
          builder=self,
          # This is the description that will appear on the datasets page.
          description=("This is the random dataset. "
                      "inputs are vectors produced by random gaussians, "
                      "and outputs generated from a random mutinomial distribution."),
          # tfds.features.FeatureConnectors
          features=tfds.features.FeaturesDict({
              "input": tfds.features.Tensor(shape=(RandomGaussian.SEQ_LEN,RandomGaussian.INPUT_DIM,),dtype=tf.float32),
              "label": tfds.features.ClassLabel(num_classes=RandomGaussian.NUM_CLASSES),
          }),
          # If there's a common (input, target) tuple from the features,
          # specify them here. They'll be used if as_supervised=True in
          # builder.as_dataset.
          supervised_keys=("input", "label"),
      )

  def _split_generators(self, dl_manager):
    # Specify the splits
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
            },
        ),
    ]

  def _generate_examples(self):
    # Yields examples from the dataset
    for inp_id in np.arange(RandomGaussian.NUMBER_OF_EXAMPLES):
      yield inp_id, {
          "input": tf.random.normal(shape=(RandomGaussian.SEQ_LEN, RandomGaussian.INPUT_DIM,), dtype=tf.float32),
          "label": np.random.randint(RandomGaussian.NUM_CLASSES),
      }