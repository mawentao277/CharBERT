export CUDA_VISIBLE_DEVICES=1,2,3
DATA_DIR=data/CoNLL2003/NER-en
MODEL_DIR=data/model/charbert-pretrain
OUTPUT_DIR=model/output/ner
python run_ner.py --data_dir ${DATA_DIR} \
                  --model_type bert \
                  --model_name_or_path $MODEL_DIR \
                  --output_dir ${OUTPUT_DIR} \
                  --num_train_epochs 3 \
                  --learning_rate 3e-5 \
                  --char_vocab ./data/dict/bert_char_vocab \ #./data/dict/roberta_char_vocab for robera
                  --per_gpu_train_batch_size 6 \
                  --do_train \
                  --do_predict \
                  --overwrite_output_dir
