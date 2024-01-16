#!/bin/bash

SPLIT=balance_split
SAMPLE_RATIO=50
CROP_SIZE=250
DATA_TYPE=random-${SAMPLE_RATIO}_crop-${CROP_SIZE}
ROOT_DIR=. # TODO: change this
OUT_DIR=pointgroup_data
NUM_PROC=20

# split data
python split_data.py --random_sample_ratio $SAMPLE_RATIO --train_crop_size $CROP_SIZE --val_crop_size $CROP_SIZE

# preprocessing
EVAL_TYPE=val_val_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC

EVAL_TYPE=val_test_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC

EVAL_TYPE=train_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC
