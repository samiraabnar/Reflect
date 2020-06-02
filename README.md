This project is called **Reflect**!

Reflect aims at exploring the power of knowledge distillation in transfering inductive biases of the teacher model to the student model! 
To read more about our findings in this project checkout this blog post on ["Distilling Inductive Biases"](https://samiraabnar.github.io/articles/2020-05/indist), or our paper on ["Transferring Inductive Biases Through Knowledge Distillation"](https://arxiv.org/abs/2006.00555).


#### How to use our codes to train a model through distillation:
```
python distill/distill_main.py \
--task=mnist \
--teacher_model=cl_vff \
--student_model=cl_vff \
--student_exp_name=gc_f_std300 \
--teacher_exp_name=gc_o_tchr300 \
--teacher_config=ff_mnist4 \
--student_config=ff_mnist4 \
--distill_mode=offline \
--batch_size=128 \
--keep_some_checkpoints=True \
--max_checkpoints=15 \
--distill_config=pure_dstl5_4_crs_slw_3 
```

#### How to use our  code to  train a model independently:
* For image processing  models:
```
  python mnist_trainer.py \
  --model=cl_vff \
  --task=mnist \
  --model_config=ff_mnist4 \
  --train_config=adam_mid \
  --batch_size=128 \
  --exp_name=trial1
```
* For language processing models:
```
  python keras_trainer.py \
  --model=lm_lstm_shared_emb \
  --task=word_sv_agreement_lm \
  --model_config=lstm_drop31_v3 \
  --train_config=radam_slw2 \
  --batch_size=512 \
  --exp_name=trial1
```

* Evaluation and analysis scrips can be found under the notebook directory. 

------------- 
This repo borrows and adapts code from:
1. The [Transformers library of HuggingFace](https://github.com/huggingface/transformers)
2. [TalLinzen/rnn_agreement repository](https://github.com/TalLinzen/rnn_agreement)
