export CUDA_VISIBLE_DEVICES=1,2,3
source ~/.bashrc
model_name_or_path=data/model/charbert-pretrain
SQUAD2_DIR=data/squad
OUTPUT_DIR=model/output/squad
python run_squad.py \
    --model_type bert \
    --model_name_or_path ${model_name_or_path} \
    --do_train \
    --do_eval \
    --data_dir $SQUAD2_DIR \
    --train_file $SQUAD2_DIR/train-v1.1.json \
    --predict_file $SQUAD2_DIR/dev-v1.1.json \
    --char_vocab ./data/dict/bert_char_vocab \ #./data/dict/roberta_char_vocab for robera
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 2000 \
    --max_seq_length 384 \
    --overwrite_output_dir \
    --doc_stride 128 \
    --output_dir ${OUTPUT_DIR}
