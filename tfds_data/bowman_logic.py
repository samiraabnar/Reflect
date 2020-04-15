import tensorflow_datasets as tfds
import tensorflow as tf
import os
import glob
from util import constants
import numpy as np

class BowmanLogic(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.1.0')


  def __init__(self, train_n=6, test_n=12 ,**kwargs):
    super(BowmanLogic, self).__init__(**kwargs)
    self.train_n = train_n
    self.test_n = test_n



  def _info(self):
    self.text_encoder_config = tfds.features.text.TextEncoderConfig(
      encoder=tfds.features.text.TokenTextEncoder(vocab_list=vocab_list,
                                                  additional_tokens=constants.all))

    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("Prepositional Logic of Sam Bowman"),
      # tfds.features.FeatureConnectors
      features=tfds.features.FeaturesDict({
        "statement_a": tfds.features.Text(
          encoder_config=self.text_encoder_config),
        "statement_b": tfds.features.Text(
          encoder_config=self.text_encoder_config),
        "relation": tfds.features.ClassLabel(names=['=', '<', '>', '|', '#', '^', 'v']),
        "n_ops": tf.int32,
      }),
      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      #supervised_keys=("sentence", "verb_class"),
    )

  def _vocab_text_gen(self, input_file):
    for _, ex in self._generate_examples(input_file):
      yield ex["statement_a"] + ' ' + ex["statement_b"]

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs


    extracted_dir = 'data/bowman_logic'
    # Specify the splits
    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "input_file_path": os.path.join(extracted_dir, "train*"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs={
          "input_file_path": os.path.join(extracted_dir, "valid*"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "input_file_path": os.path.join(extracted_dir, "test*"),
        },
      ),
    ]

  def _generate_examples(self, input_file_path):
    """ Yields examples from the dataset

    :param input_file_path:
    :return: example
    """
    example_id = 0
    for file in glob.glob(input_file_path):
      n = int(''.join(filter(str.isdigit, file)))
      print("generate examples: ", file)
      for example in open(file, 'r'):
        col = example.split('\t')
        y = col[0]
        xs1 = col[1]
        xs2 = col[2]
        example_id += 1
        yield example_id, {
          "statement_a": xs1,
          "statement_b": xs2,
          "relation": y,
          "n_ops": n
        }

  def sentence_encoder(self):
    return self.info.features["statement_a"].encoder

  def vocab_size(self):
    """Retrieves the dictionary mapping word indices back to words.
    Arguments:
        path: where to cache the data (relative to `~/.keras/dataset`).
    Returns:
        The word index dictionary.
    """
    return self.info.features["statement_a"].encoder.vocab_size

def build_vocab(data_dir='data/bowman_logic', input_file_path='data/bowman_logic/'):
  vocab_list = []
  for file in glob.glob(input_file_path+"train*"):
    for line in open(file, 'r'):
      col = line.split('\t')
      xs1 = col[1]
      xs2 = col[2]
      example = ' '.join([xs1, xs2])
      for word in example.split():
        if word not in vocab_list:
          vocab_list.append(word)

  for file in glob.glob(input_file_path+"test*"):
    for line in open(file, 'r'):
      col = line.split('\t')
      xs1 = col[1]
      xs2 = col[2]
      example = ' '.join([xs1, xs2])
      for word in example.split():
        if word not in vocab_list:
          vocab_list.append(word)

  for file in glob.glob(input_file_path+"valid*"):
    for line in open(file, 'r'):
      col = line.split('\t')
      xs1 = col[1]
      xs2 = col[2]
      example = ' '.join([xs1, xs2])
      for word in example.split():
        if word not in vocab_list:
          vocab_list.append(word)

  np.save(os.path.join(data_dir, 'vocab'), vocab_list)


if __name__ == '__main__':
  build_vocab()
  # bl = BowmanLogic(data_dir='data')
  # bl.download_and_prepare(download_dir='tmp/',
  #                                  download_config=tfds.download.DownloadConfig(register_checksums=True))
  #
  # dataset = bl.as_dataset(split="validation")
  # dataset = tfds.as_numpy(dataset)
  # for example in dataset:
  #   print(example)
  #   break


