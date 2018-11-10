#!/bin/bash

~/python edit/train.py \
	-save_path projects/ins_del_edit/model \
	-log_home projects/ins_del_edit/model \
	-online_process_data \
	-train_src projects/ins_del_edit/train.src \
	-train_tgt projects/ins_del_edit/train.tgt \
	-layers 1 -enc_rnn_size 512 -brnn -word_vec_size 300 -dropout 0.5 \
	-batch_size 64 -beam_size 1 \
	-epochs 20 \
	-gpus 0 \
	-optim adam -learning_rate 0.001 \
	-curriculum 0 -extra_shuffle \
	-start_eval_batch 15000 -eval_per_batch 1200 -seed 12345 -cuda_seed 12345 -log_interval 100

