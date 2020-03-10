import string
from collections import Counter

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
from tensorflow_datasets.text import Lm1bConfig
from prep_data.build_dictionary import build_and_save_dic
from util import text_util, constants
from util.text_util import deps_from_tsv, deps_to_tsv


def build_dic(data_dir):
  worddict = {}
  reserved = [constants.pad, constants.unk, constants.bos, constants.eos]
  worddict[constants.pad] = constants.pad_idx
  worddict[constants.unk] = constants.unk_idx
  worddict[constants.bos] = constants.bos_idx
  worddict[constants.eos] = constants.eos_idx

  text_encoder_config = tfds.features.text.TextEncoderConfig(
    encoder_cls=tfds.features.text.SubwordTextEncoder,
    vocab_size=2 ** 13)

  feature = tfds.features.Text(
    encoder_config=text_encoder_config)

  databuilder = tfds.text.lm1b.Lm1b(data_dir='data')
  databuilder.download_and_prepare(download_dir='tmp/',
                                   download_config=tfds.download.DownloadConfig(register_checksums=True))

  def generator():
    dataset = databuilder.as_dataset(split="train")
    dataset = tfds.as_numpy(dataset)
    for sentence in dataset:
      yield sentence['text']

  encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_generator=generator(),
    target_vocab_size=2 ** 13,
    max_subword_length=20,
    max_corpus_chars=None,
    reserved_tokens=reserved)

  encoder.save_to_file(filename_prefix=os.path.join(data_dir,'sublmvocab'))


if __name__ == '__main__':
  data_dir = 'data/lm1b/'
  text_encoder_config = tfds.features.text.TextEncoderConfig(
    encoder=tfds.features.text.SubwordTextEncoder.load_from_file(os.path.join(data_dir,'sublmvocab'))
  )

  config = Lm1bConfig(
    old_version='1.0.0',
    text_encoder_config=text_encoder_config,
    name='subwords'
  )

  databuilder = tfds.text.lm1b.Lm1b(data_dir='data',
                                    config=config)
  databuilder.download_and_prepare(download_dir='tmp/',
                                    download_config=tfds.download.DownloadConfig(register_checksums=True))


  dataset = databuilder.as_dataset(split="test", batch_size=1000)
  dataset = tfds.as_numpy(dataset)
  for batch in dataset:
    print(batch['text'][0])
    print(databuilder.info.features["text"].encoder.decode(batch['text'][0]))
    print(databuilder.info.features["text"].encoder.encode('<bos> hi <eos>'))
    print(databuilder.info.features["text"].encoder.encode('<bos>'))
    print(databuilder.info.features["text"].encoder.encode('<eos>'))
    print(databuilder.info.features["text"].encoder.encode('<pad>'))


    break
  #
  # print(databuilder.info.features.keys())
  # print(databuilder.info.features['text'].vocab_size)
  #
  #
  #
