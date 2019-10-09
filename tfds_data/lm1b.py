from collections import Counter

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np

from prep_data.build_dictionary import build_and_save_dic
from util import text_util, constants
from util.text_util import deps_from_tsv, deps_to_tsv


if __name__ == '__main__':

  databuilder = tfds.text.lm1b.Lm1b(data_dir='data')
  databuilder.download_and_prepare(download_dir='tmp/',
                                    download_config=tfds.download.DownloadConfig(register_checksums=True))


  dataset = databuilder.as_dataset(split="test", batch_size=1000)
  dataset = tfds.as_numpy(dataset)
  for batch in dataset:
    print(batch)

    break

  print(databuilder.vocab_size())


