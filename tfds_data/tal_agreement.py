import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

from prep_data.build_dictionary import build_and_save_dic
from util import text_util, constants
from util.text_util import deps_from_tsv, deps_to_tsv


class SVAgreement(tfds.core.GeneratorBasedBuilder):
  """ This is the dataset for evaluating the ability of language models to learn syntax.
  Paper:
  Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies
  Tal Linzen, Emmanuel Dupoux, Yoav Goldberg
  """

  VERSION = tfds.core.Version('0.1.0')

  CLASS_TO_CODE = {'VBZ': 0, 'VBP': 1}
  CODE_TO_CLASS = {x: y for y, x in CLASS_TO_CODE.items()}

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("This is the dataset for subject verb agreement "
                   "to assess the ability of language models to learn syntax"),
      # tfds.features.FeatureConnectors
      features=tfds.features.FeaturesDict({
        "sentence": tfds.features.Text(
          encoder_config=tfds.features.text.TextEncoderConfig(
          encoder_cls=tfds.features.text.SubwordTextEncoder,
          vocab_size=2**13)),
        # Here, labels can be of 5 distinct values.
        "verb_class": tfds.features.ClassLabel(names=["VBZ", "VBP"]),
        "verb_position": tf.int32,
      }),
      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=("sentence", "verb_class"),
      # Homepage of the dataset for documentation
      urls=["https://github.com/TalLinzen/rnn_agreement"],
      # Bibtex citation for the dataset
      citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Linzen, Tal; Dupoux,Emmanuel; Goldberg, Yoav},"}""",
    )

  def _vocab_text_gen(self, input_file):
    for _, ex in self._generate_examples(input_file):
      yield ex["sentence"]

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    extracted_path = dl_manager.download_and_extract(
      'http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz')

    def make_splits(extracted_path, data_dir, prop_train=0.1, prop_valid=0.01):

      # for reproducibility
      np.random.seed(42)
      print('| read in the data')
      data = deps_from_tsv(extracted_path)
      print('| shuffling')
      np.random.shuffle(data)

      n_train = int(len(data) * prop_train)
      n_valid = int(len(data) * prop_valid)
      train = data[:n_train]
      valid = data[n_train: n_train + n_valid]
      test = data[n_train + n_valid:]

      print('| splitting')
      deps_to_tsv(train, os.path.join(data_dir, "train.tsv"))
      deps_to_tsv(valid, os.path.join(data_dir, "valid.tsv"))
      deps_to_tsv(test,  os.path.join(data_dir, "test.tsv"))
      print('| done!')
    make_splits(extracted_path,self.data_dir)

    # Generate vocabulary from training data if SubwordTextEncoder configured
    self.info.features["sentence"].maybe_build_from_corpus(
      self._vocab_text_gen(os.path.join(self.data_dir, "train.tsv")))

    # Specify the splits
    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "input_file_path": os.path.join(self.data_dir, "train.tsv"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs={
          "input_file_path": os.path.join(self.data_dir, "valid.tsv"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "input_file_path": os.path.join(self._data_dir, "test.tsv"),
        },
      ),
    ]

  def _generate_examples(self, input_file_path):
    """ Yields examples from the dataset

    :param input_file_path:
    :return: example
    """

    # Read the input data out of the source files
    data = deps_from_tsv(input_file_path)

    # And yield examples as feature dictionaries
    example_id = 0
    for example in data:
      example_id += 1
      yield example_id, {
        "sentence": example['sentence'],
        "verb_class": example['verb_pos'],
        "verb_position": int(example['verb_index']) - 1
      }

  def sentence_encoder(self):
    return self.info.features["sentence"].encoder

  def vocab_size(self):
    """Retrieves the dictionary mapping word indices back to words.
    Arguments:
        path: where to cache the data (relative to `~/.keras/dataset`).
    Returns:
        The word index dictionary.
    """
    return self.info.features["sentence"].encoder.vocab_size

if __name__ == '__main__':

  databuilder = SVAgreement(data_dir='data')
  databuilder.download_and_prepare(download_dir='tmp/',
                                    download_config=tfds.download.DownloadConfig(register_checksums=True))


  dataset = databuilder.as_dataset(split="validation", batch_size=1000)
  dataset = tfds.as_numpy(dataset)
  for batch in dataset:
    print("encoded_sentence:", batch['sentence'][0])
    print("decoded_sentence:", databuilder.sentence_encoder().decode(batch['sentence'][0]))
    print("verb class:", batch['verb_class'][0])
    print("verb position:",batch['verb_position'][0])
    break

  print(databuilder.vocab_size())


