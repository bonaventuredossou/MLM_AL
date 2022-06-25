#!/bin/bash

export CUDA_AVAILABLE_DEVICES=0
export PYTHONPATH=$PWD

experiment_name=active_learning

model_path="/home/femipancrace_dossou/MLM_AL/experiments_500k/active_learning_lm"
tokenizer_path="/home/femipancrace_dossou/MLM_AL/tokenizer_250k" # specify tokenizer path
ner_dataset="/home/femipancrace_dossou/MLM_AL/ner_data"

ner_model_path="${experiment_name}_ner_model"

mkdir $ner_model_path

cp $model_path/pytorch_model.bin $PWD/$ner_model_path/
cp $model_path/config.json $PWD/$ner_model_path/

MAX_LENGTH=164
MODEL_PATH=$ner_model_path
BATCH_SIZE=16
NUM_EPOCHS=50
SAVE_STEPS=1000000
TOK_PATH=$tokenizer_path

declare -a arr=("kin" "lug" "luo" "pcm" "amh" "hau" "ibo" "swa" "wol" "yor")

for SEED in 1
do
    output_dir="${experiment_name}_ner_results_${SEED}"
    mkdir $output_dir

    for i in "${arr[@]}"
    do
        OUTPUT_DIR=$output_dir/"$i"
        DATA_DIR=$ner_dataset/"$i"
        python ner_scripts/train_ner.py --data_dir $DATA_DIR \
        --model_type nil \
        --model_name_or_path $MODEL_PATH \
        --tokenizer_path $TOK_PATH \
        --output_dir $OUTPUT_DIR \
        --max_seq_length $MAX_LENGTH \
        --num_train_epochs $NUM_EPOCHS \
        --per_gpu_train_batch_size $BATCH_SIZE \
        --per_gpu_eval_batch_size $BATCH_SIZE \
        --save_steps $SAVE_STEPS \
        --seed $SEED \
        --do_train \
        --do_eval \
        --do_predict

    done
done