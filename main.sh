# aggregate the data preprocess, train and eval

export RUN_NAME=resampled_frombm25_shuffled
export GS_FOLDER=gs://crys-west4-a/t5/${RUN_NAME}
DATA_DIR=/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/${RUN_NAME}

#######################
# prepare data
#######################
export OUTP_DATA_FN=${DATA_DIR}/query_doc_pairs.train.shuffled.tsv
python pygaggle/data/create_msmarco_t5_training_pairs.py \
  --triples_train ${DATA_DIR}/triples.train.small.id.shuffled.tsv \
  --output_to_t5 ${OUTP_DATA_FN}
gsutil cp ${OUTP_DATA_FN} ${GS_FOLDER}/


#######################
# train
#######################
export MODEL_NAME=base
export PROJECT_NAME=crystina-project
export TPU_NAME=crys-1

export MODEL_INIT_CKPT=999900
# Copy pre-trained checkpoint to our target model
echo "model_checkpoint_path: \"model.ckpt-${MODEL_INIT_CKPT}\"" > checkpoint
gsutil cp checkpoint ${GS_FOLDER}
gsutil cp gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}* ${GS_FOLDER}


t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT_NAME}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="${GS_FOLDER}" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/${MODEL_NAME}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = '${GS_FOLDER}/query_doc_pairs.train.shuffled.tsv' " \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1100000" \
  --gin_param="run.save_checkpoints_steps = 10000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)"



#######################
# inference
#######################
export MODEL_DIR=${GS_FOLDER}
export DEV_DATA_FOLDER=gs://crys-west4-a/t5/data/eval_data

for ITER in 000 001 002 003 004 005 006 007 008
do
  echo "Running iter: $ITER"
  t5_mesh_transformer \
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
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1"
done



export DATA_DIR=data/msmarco_passage
export RESULT_DIR=data/msmarco_passage/results/${RUN_NAME}
mkdir -p $RESULT_DIR

gsutil cp ${GS_FOLDER}/query_doc_pair_scores.dev.small.txt*-1100000 ${RESULT_DIR}/
cat ${RESULT_DIR}/query_doc_pair_scores.dev.small.txt*-1100000 > ${RESULT_DIR}/query_doc_pair_scores.dev.small.txt


python pygaggle/data/convert_monot5_output_to_msmarco_run.py \
        --t5_output ${RESULT_DIR}/query_doc_pair_scores.dev.small.txt \
        --t5_output_ids data/msmarco_passage/query_doc_pair_ids.dev.small.tsv \
        --mono_run ${RESULT_DIR}/run.monot5_${MODEL_NAME}.dev.tsv && echo "converted"

python tools/scripts/msmarco/msmarco_passage_eval.py \
  data/msmarco_passage/dev_set/qrels.dev.small.tsv ${RESULT_DIR}/run.monot5_${MODEL_NAME}.dev.tsv
