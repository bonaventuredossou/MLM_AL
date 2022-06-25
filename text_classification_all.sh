#!/bin/bash

export CUDA_AVAILABLE_DEVICES=0

experiment_name=active_learning

model_path="/home/femipancrace_dossou/MLM_AL/experiments_500k/active_learning_lm"
tokenizer_path="/home/femipancrace_dossou/MLM_AL/tokenizer_250k" # specify tokenizer path

MODEL_PATH=$model_path
TOK_PATH=$tokenizer_path
HAUSA_DATA="/home/femipancrace_dossou/MLM_AL/hausa_classification_data"
YORUBA_DATA="/home/femipancrace_dossou/MLM_AL/yoruba_classification_data"
YOSM_DATA="/home/femipancrace_dossou/MLM_AL/yosm"
NAIJA_YOSM_DATA="/home/femipancrace_dossou/MLM_AL/naija_senti_yosm"

# Evaluate on Text Classification

export PYTHONPATH=$PWD

for SEED in 1 2 3 4 5
do 

    output_dir=/home/femipancrace_dossou/MLM_AL/classification_results/"${MODEL_PATH}_hausa_${SEED}"
    python classification_scripts/classification_trainer.py --data_dir $HAUSA_DATA \
    --model_dir $MODEL_PATH \
    --tok_dir $TOK_PATH \
    --output_dir $output_dir \
    --language hausa \
    --seed $SEED \
    --max_seq_length 256

    output_dir=/home/femipancrace_dossou/MLM_AL/classification_results/"${MODEL_PATH}_yoruba_${SEED}"
    python classification_scripts/classification_trainer.py --data_dir $YORUBA_DATA \
    --model_dir $MODEL_PATH \
    --tok_dir $TOK_PATH \
    --output_dir $output_dir \
    --language yoruba \
    --seed $SEED \
    --max_seq_length 256

    output_dir=/home/femipancrace_dossou/MLM_AL/classification_results/"${MODEL_PATH}_YOSM_${SEED}"
    python classification_scripts/classification_sentiment.py --data_dir $YOSM_DATA \
    --model_dir $MODEL_PATH \
    --tok_dir $TOK_PATH \
    --output_dir $output_dir \
    --language yoruba \
    --seed $SEED \
    --max_seq_length 256

    output_dir=/home/femipancrace_dossou/MLM_AL/classification_results/"${MODEL_PATH}_NAIJA_YOSM_${SEED}"
    python classification_scripts/classification_sentiment.py --data_dir $NAIJA_YOSM_DATA \
    --model_dir $MODEL_PATH \
    --tok_dir $TOK_PATH \
    --output_dir $output_dir \
    --language yoruba \
    --seed $SEED \
    --max_seq_length 256

done