# ResponseEdit
Resources of our paper at AAAI-19 ``Response Generation by Context-aware Prototype Editing"

## Data Format
test.txt and dev.txt are formatted as 

"current context \t current response \t prototype context \t prototype response" .

As the input format of our code is "prototype context \t ins words \t del words", we further provide test_format.txt and dev_format.txt. Training dataset follows the same rule. File projects/train.src shows the format of input. 

## Code
train.py and console_translate.py in the edit folder are the entires of the training and testing process. Please write 
```
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
```
for training. You could learn details of hyper-parameters in the xargs.py

Translate command is 
```
~/python edit/console_translate.py
-model projects/ins_del_edit/model/model_e9.pt \
-gpu 1 \
-src projects/ins_del_edit/train.src
```

We evaluate the final result with NLG-EVAL(https://github.com/Maluuba/nlg-eval)

## Acknowledgement
A large part of this code is borrowed from Open-NMT-Pytorch. 

## Reference 
Please cite our paper if you use related resource of our code. 
```
@article{wu2018response,
  title={Response Generation by Context-aware Prototype Editing},
  author={Wu, Yu and Wei, Furu and Huang, Shaohan and Li, Zhoujun and Zhou, Ming},
  journal={arXiv preprint arXiv:1806.07042},
  year={2018}
}
```
