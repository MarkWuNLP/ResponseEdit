# ResponseEdit
Resources of our paper at AAAI-19 ``Response Generation by Context-aware Prototype Editing" [link](https://arxiv.org/abs/1806.07042)

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
## Requirement
Pytorch >= 0.4

The devlopment environment of the project is windows, so some minor changes may be required when running with Linux. 
## Acknowledgement
A large part of this code is borrowed from Open-NMT-Pytorch. 

## Data Format
train, dev and test data are formatted as 

"current context \t current response \t prototype context \t prototype response" .

As the input format of our code is "prototype context \t ins words \t del words", you should further format data like that. File projects/train.src shows the format of input. Please email us if you need the dataset. 

##Baseline
Yunli Wang, a Master student of Beihang University, helped me to implement the method proposed in "Two are Better than One: An Ensemble of Retrieval- and Generation-Based Dialog Systems" (https://arxiv.org/abs/1610.07149). The source code is uploaded to https://github.com/jimth001/Bi-Seq2Seq
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
## Note 
It seems that the code does not work when layer num is not 1. 
