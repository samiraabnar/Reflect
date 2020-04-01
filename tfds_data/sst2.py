import tensorflow_datasets as tfds

from util import constants


class Sst2(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.1.0')


  def __init__(self,**kwargs):
    super(Sst2, self).__init__(name='sst2',**kwargs)

  def _info(self):
    self.text_encoder_config = tfds.features.text.TextEncoderConfig(
      encoder_cls=tfds.features.text.SubwordTextEncoder,
      vocab_size=2 ** 13)

    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("Glue SST2"),

      features=tfds.features.FeaturesDict({
        "sentence": tfds.features.Text(
          encoder_config=self.text_encoder_config),
        "label": tfds.features.ClassLabel(names=['+','-']),
      }),
      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=("sentence", "label"),
    )

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    examples = tfds.load('glue/sst2')
    train_examples, val_examples = examples['train'], examples['validation']
    self.info.features["sentence"].maybe_build_from_corpus((example['sentence'].numpy() for example in train_examples),
                                                           reserved_tokens=[constants.pad, constants.unk, constants.bos, constants.eos])

    # Specify the splits
    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "input_file_path": "train",
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs={
          "input_file_path": "validation",
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "input_file_path": "test",
        },
      ),
    ]

  def _generate_examples(self, input_file_path):
    """ Yields examples from the dataset

    :param input_file_path:
    :return: example
    """
    examples = tfds.load(name='glue/sst2', split=input_file_path)

    for example in examples:
      yield example['idx'], {'label': example['label'].numpy(),
              'sentence': example['sentence'].numpy()
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

  databuilder = Sst2(data_dir='data')
  databuilder.download_and_prepare(download_dir='tmp/',
                                    download_config=tfds.download.DownloadConfig(register_checksums=True))


  dataset = databuilder.as_dataset(split="validation", batch_size=1000)
  dataset = tfds.as_numpy(dataset)
  for batch in dataset:
    print("encoded_sentence:", batch['sentence'])
    print("decoded_sentence:", databuilder.sentence_encoder().decode(batch['sentence'][0]))
    print("label:",batch['label'][0])

    break

  print(databuilder.vocab_size())