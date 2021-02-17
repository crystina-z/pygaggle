# export GS_FOLDER=gs://crys-west4-a/t5/replicate 
# export GS_FOLDER=gs://crys-west4-a/t5/resampled_fromindri
# export GS_FOLDER=gs://crys-west4-a/t5/resampled_frombm25
# export GS_FOLDER=gs://crys-west4-a/t5/resampled_fromindri_shuffled 
export GS_FOLDER=gs://crys-west4-a/t5/resampled_frombm25_shuffled 

export MODEL_NAME=base
export PROJECT_NAME=crystina-project
export TPU_NAME=crys-1 
export MODEL_INIT_CKPT=999900 

# Copy pre-trained checkpoint to our target model
echo "model_checkpoint_path: \"model.ckpt-${MODEL_INIT_CKPT}\"" > checkpoint
gsutil cp checkpoint ${GS_FOLDER}
gsutil cp gs://t5-data/pretrained_models/${MODEL_NAME}/model.ckpt-${MODEL_INIT_CKPT}* ${GS_FOLDER}

outp_fn=out.log_exp

nohup t5_mesh_transformer  \
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
  --gin_param="tsv_dataset_fn.filename = 'gs://crys-west4-a/t5/resampled_frombm25_shuffled/query_doc_pairs.train.shuffled.tsv' " \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1100000" \
  --gin_param="run.save_checkpoints_steps = 10000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  >>  $outp_fn 2>&1 &

tail -100f $outp_fn 

# --gin_param="tsv_dataset_fn.filename = 'gs://crys-west4-a/t5/resampled_fromindri_shuffled/query_doc_pairs.train.shuffled.tsv' " \
# --gin_param="tsv_dataset_fn.filename = 'gs://crys-west4-a/t5/resampled_frombm25/query_doc_pairs.train.tsv' " \
# --gin_param="tsv_dataset_fn.filename = 'gs://crys-west4-a/t5/resampled/data/resampled.training.tsv' " \
