export MODEL_NAME=base
export PROJECT_NAME=crystina-project
export TPU_NAME=crys-1 
# export GS_FOLDER=gs://crys-west4-a/t5/replicate 
# export GS_FOLDER=gs://crys-west4-a/t5/resampled_frombm25
# export GS_FOLDER=gs://crys-west4-a/t5/resampled_fromindri_shuffled
export GS_FOLDER=gs://crys-west4-a/t5/resampled_frombm25_shuffled

# export MODEL_DIR=gs://castorini/monot5/experiments/${MODEL_NAME}
# export MODEL_DIR=gs://crys-west4-a/t5/resampled
# export MODEL_DIR=gs://crys-west4-a/t5/replicate
export MODEL_DIR=${GS_FOLDER}

# Copy pre-trained checkpoint to our target model
# echo "model_checkpoint_path: \"model.ckpt-${MODEL_INIT_CKPT}\"" > checkpoint
# gsutil cp checkpoint ${GS_FOLDER}
# gsutil cp gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}* ${GS_FOLDER}

export DEV_DATA_FOLDER=gs://crys-west4-a/t5/data/eval_data 

for ITER in 000 001 002 003 004 005 006 007 008 
do
  echo "Running iter: $ITER" >> out.log_eval_exp
  nohup t5_mesh_transformer \
    --tpu="${TPU_NAME}" \
    --gcp_project="${PROJECT_NAME}" \
    --tpu_zone="europe-west4-a" \
    --model_dir="${MODEL_DIR}" \
    --gin_file="gs://t5-data/pretrained_models/${MODEL_NAME}/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
    --gin_param="infer_checkpoint_step = 1100000" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
    --gin_param="Bitransformer.decode.max_decode_length = 2" \
    --gin_param="input_filename = '${DEV_DATA_FOLDER}/query_doc_pairs.dev.small.txt${ITER}'" \
    --gin_param="output_filename = '${GS_FOLDER}/query_doc_pair_scores.dev.small.txt${ITER}'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
    >> out.log_eval_exp 2>&1
done &

tail -100f out.log_eval_exp

