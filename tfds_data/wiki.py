import tensorflow_datasets as tfds
from nltk import tokenize
import nltk
nltk.download('punkt')

class WikiEn(tfds.core.GeneratorBasedBuilder):

  VERSION = tfds.core.Version('0.1.0')


  def __init__(self,**kwargs):
    super(WikiEn, self).__init__(**kwargs)

  def _info(self):
    self.text_encoder_config = tfds.features.text.TextEncoderConfig(
      encoder_cls=tfds.features.text.SubwordTextEncoder,
      vocab_size=2 ** 13)

    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("wiki en"),

      features=tfds.features.FeaturesDict({
        "sentence": tfds.features.Text(
          encoder_config=self.text_encoder_config),
        "title": tfds.features.Text(encoder_config=self.text_encoder_config),
      }),
      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=("sentence", "title"),
    )

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    examples = tfds.load('wikipedia/20190301.en', split='train')
    self.info.features["sentence"].maybe_build_from_corpus((example['text'].numpy().decode('utf-8') for example in examples))

    # Specify the splits
    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "input_file_path": "train[:98%]",
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs={
          "input_file_path": "train[98%:99%]",
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "input_file_path": "train[99%:]",
        },
      ),
    ]

  def _generate_examples(self, input_file_path):
    """ Yields examples from the dataset

    :param input_file_path:
    :return: example
    """
    examples = tfds.load(name='wikipedia/20190301.en', split=input_file_path)

    example_id = 0
    for example in examples:
      paragprahs = example['text'].numpy().decode('utf-8').split('\n')
      title = example['title'].numpy().decode('utf-8')
      for p in paragprahs:
        sentences = map(lambda s: s.strip(), tokenize.sent_tokenize(p))
        sentences = filter(lambda s: len(s.split()) > 2 and len(s) < 1000, sentences)
        for s in sentences:
          example_id += 1
          print(title, s)
          yield example_id, {'sentence': s,
                 'title': title}

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

  databuilder = WikiEn(data_dir='data')
  databuilder.download_and_prepare(download_dir='tmp/',
                                    download_config=tfds.download.DownloadConfig(register_checksums=True))


  dataset = databuilder.as_dataset(split="validation", batch_size=100)
  dataset = tfds.as_numpy(dataset)
  for batch in dataset:
    print("encoded_sentence:", batch['sentence'])
    print("decoded_sentence:", databuilder.sentence_encoder().decode(batch['sentence'][0]))
    print("title:",batch['title'][0])

    break

  print(databuilder.vocab_size())