#!/bin/bash

SPLIT=balance_split
DATA_TYPE=random-50_crop-250
ROOT_DIR=. # TODO: change this
OUT_DIR=pointgroup_data
NUM_PROC=20

# split data
python split_data.py --random_sample_ratio 50 --train_crop_size 50 --val_crop_size 250

# preprocessing
EVAL_TYPE=val_val_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC

EVAL_TYPE=val_test_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC

EVAL_TYPE=train_250m
python prepare_data.py --split_type $SPLIT --data_type $DATA_TYPE --eval_type $EVAL_TYPE --root_dir $ROOT_DIR --output_dir $OUT_DIR --num_processes $NUM_PROC
