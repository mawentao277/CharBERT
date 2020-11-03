export CUDA_VISIBLE_DEVICES=1,2
export GLUE_DIR=data/glue_dataset/QNLI
export MODEL_DIR=data/model/charbert-pretrain
export TASK_NAME=QNLI
OUTPUT_DIR=model/output/qnli

python run_glue.py \
             --model_name_or_path ${MODEL_DIR} \
             --task_name ${TASK_NAME} \
             --model_type bert \
             --do_train \
             --do_eval \
             --data_dir ${GLUE_DIR} \
             --max_seq_length 128 \
             --per_gpu_eval_batch_size=16 \
             --per_gpu_train_batch_size=16 \
             --char_vocab ./data/dict/bert_char_vocab \
             --learning_rate 3e-5 \
             --save_steps 1000 \
             --num_train_epochs 8.0 \
             --overwrite_output_dir \
             --output_dir ${OUTPUT_DIR}
