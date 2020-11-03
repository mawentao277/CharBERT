export CUDA_VISIBLE_DEVICES=1,2,3
DATA_DIR=data
MODEL_DIR=data/model/bert_base_cased #pretrain charbert by bert_base_cased model
OUTPUT_DIR=model/output/mlm
python3 run_lm_finetuning.py \
    --model_type bert \
    --model_name_or_path ${MODEL_DIR} \
    --do_train \
    --do_eval \
    --train_data_file $DATA_DIR/testdata/mlm_pretrain_enwiki.train.t \
    --eval_data_file $DATA_DIR/testdata/mlm_pretrain_enwiki.test.t \
    --term_vocab ${DATA_DIR}/dict/term_vocab \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --char_vocab ./data/dict/bert_char_vocab \ #./data/dict/roberta_char_vocab for robera
    --mlm_probability 0.10 \
    --input_nraws 1000 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 10000 \
    --block_size 384 \
    --overwrite_output_dir \
    --mlm \
    --output_dir ${OUTPUT_DIR}
