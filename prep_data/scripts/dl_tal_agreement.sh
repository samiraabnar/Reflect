#!/bin/bash

# create data dir
DATA_DIR=data/tal_agreement
mkdir DATA_DIR

DATA_FILE=http://tallinzen.net/media/rnn_agreement/agr_50_mostcommon_10K.tsv.gz

wget -P $DATA_DIR $DATA_FILE
gunzip $DATA_DIR/agr_50_mostcommon_10K.tsv.gz

echo "done!"