export CUDA_VISIBLE_DEVICES=1,2,3
DATA_DIR=data/PTB/postags
MODEL_DIR=data/model/charbert-pretrain
OUTPUT_DIR=model/output/pos

python run_pos.py --data_dir ${DATA_DIR} \
                  --model_type bert \
                  --model_name_or_path $BERT_MODEL \
                  --output_dir $BERT_MODEL \
                  --labels ${DATA_DIR}/pos_labels \
                  --num_train_epochs 1 \
                  --char_vocab ./data/dict/bert_char_vocab \
                  --learning_rate 2e-5 \
                  --per_gpu_train_batch_size 16 \
                  --per_gpu_eval_batch_size 16 \
                  --max_seq_length 128 \
                  --do_train \
                  --do_predict \
                  --do_eval \
                  --overwrite_output_dir
