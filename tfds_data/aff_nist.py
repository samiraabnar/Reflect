import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import glob
import scipy.io as spio


MNIST_IMAGE_SHAPE = (40, 40, 1)
MNIST_NUM_CLASSES = 10


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_affNIST(path):
  inputs = []
  labels = []
  for p in glob.glob(path):
    dataset = loadmat(p)

    labels.extend(dataset['affNISTdata']['label_int'])
    inputs.extend(dataset['affNISTdata']['image'].T)

  labels = np.asarray(labels)
  inputs = np.asarray(inputs)
  inputs = np.reshape(inputs, (-1, 40, 40))

  return inputs, labels



class AffNist(tfds.core.GeneratorBasedBuilder):
  """ This is the dataset for evaluating the ability of language models to learn syntax.
  Paper:
  Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies
  Tal Linzen, Emmanuel Dupoux, Yoav Goldberg
  """

  VERSION = tfds.core.Version('0.1.0')

  def __init__(self, **kwargs):
    super(AffNist, self).__init__(**kwargs)

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      # This is the description that will appear on the datasets page.
      description=("This is the AFFnist dataset"),
      # tfds.features.FeatureConnectors
      features=tfds.features.FeaturesDict({
        "image": tfds.features.Image(shape=(40, 40, 1)),
        "label": tfds.features.ClassLabel(num_classes=10),
      }),

      # If there's a common (input, target) tuple from the features,
      # specify them here. They'll be used if as_supervised=True in
      # builder.as_dataset.
      supervised_keys=("image", "label"),
      # Homepage of the dataset for documentation
      # Bibtex citation for the dataset
    )

  def _split_generators(self, dl_manager):
    # Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs

    #     train_extracted_path = dl_manager.download_and_extract('http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_batches.zip')
    #     test_extracted_path =  dl_manager.download_and_extract('http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test_batches.zip')
    #     valid_extracted_path =  dl_manager.download_and_extract('http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/validation_batches.zip')
    #     print(self.data_dir)

    # Specify the splits
    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs={
          "input_file_path": os.path.join("../tmp/training.mat"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs={
          "input_file_path": os.path.join("../tmp/validation.mat"),
        },
      ),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs={
          "input_file_path": os.path.join("../tmp/test.mat"),
        },
      ),
    ]

  def _generate_examples(self, input_file_path):
    """ Yields examples from the dataset

    :param input_file_path:
    :return: example
    """
    print(input_file_path)
    # Read the input data out of the source files
    images, labels = load_affNIST(input_file_path)
    print(images.shape)
    # And yield examples as feature dictionaries
    example_id = 0
    for image, label in zip(images, labels):
      example_id += 1
      yield example_id, {
        "image": image[:,None],
        "label": label
      }