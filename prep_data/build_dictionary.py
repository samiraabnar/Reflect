from util import text_util as utils
from util import constants
from sys import argv
import numpy as np
import os


def build_and_save_dic(input_file, data_dir):
    worddict = {}
    worddict[constants.pad] = constants.pad_idx
    worddict[constants.unk] = constants.unk_idx
    worddict[constants.bos] = constants.bos_idx
    worddict[constants.eos] = constants.eos_idx

    input_file = os.path.join(data_dir, input_file)
    for dep in utils.deps_from_tsv(input_file):
        for w in dep['sentence'].split():
            if w not in worddict:
                worddict[w] = len(worddict)

    vocab_file = os.path.join(data_dir, 'vocab')
    print('| write vocabulary to %s' % vocab_file)

    np.save(vocab_file, arr=worddict)

    print('| vocabulary size %d' % len(worddict))
    print('| done!')


if __name__ == '__main__':
    data_dir = argv[1]
    input_file = argv[2]

    build_and_save_dic(input_file=input_file,
                       data_dir=data_dir)