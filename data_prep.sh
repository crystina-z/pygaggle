# export DATA_DIR=data/msmarco_passage/resampled/fromindri
# DATA_DIR=data/msmarco_passage/resampled/frombm25
DATA_DIR=/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/replicate
# export GS_FOLDER=gs://crys-west4-a/t5/data
export GS_FOLDER=gs://crys-west4-a/t5/replicate_shuffled
# mkdir ${DATA_DIR}

# cd ${DATA_DIR}
# wget https://storage.googleapis.com/duobert_git/triples.train.small.tar.gz
# tar -xvf triples.train.small.tar.gz
# rm triples.train.small.tar.gz
# cd ../../

outp_fn=${DATA_DIR}/query_doc_pairs.train.shuffled.tsv
python pygaggle/data/create_msmarco_t5_training_pairs.py --triples_train ${DATA_DIR}/triples.train.small.id.shuffled.tsv --output_to_t5 ${outp_fn}
# python pygaggle/data/create_msmarco_t5_training_pairs.py --triples_train ${DATA_DIR}/resample.small.triple.fromindri.shuffled.txt --output_to_t5 ${outp_fn}
# python pygaggle/data/create_msmarco_t5_training_pairs.py --triples_train ${DATA_DIR}/resample.small.triple.frombm25.txt --output_to_t5 ${DATA_DIR}/query_doc_pairs.train.tsv
# python pygaggle/data/create_msmarco_t5_training_pairs.py --triples_train ${DATA_DIR}/triples.train.small.tsv --output_to_t5 ${DATA_DIR}/query_doc_pairs.train.tsv
gsutil cp ${outp_fn} ${GS_FOLDER}/

exit

# dev set
cd ${DATA_DIR}
wget https://storage.googleapis.com/duobert_git/run.bm25.dev.small.tsv
wget https://www.dropbox.com/s/hq6xjhswiz60siu/queries.dev.small.tsv
wget https://www.dropbox.com/s/5t6e2225rt6ikym/qrels.dev.small.tsv
wget https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz
tar -xvf collection.tar.gz
rm collection.tar.gz
mv run.bm25.dev.small.tsv run.dev.small.tsv
cd ../../


# verify 
python tools/eval/msmarco_eval.py ${DATA_DIR}/qrels.dev.small.tsv ${DATA_DIR}/run.dev.small.tsv

# convert
python pygaggle/data/create_msmarco_monot5_input.py --queries ${DATA_DIR}/queries.dev.small.tsv \
                                      --run ${DATA_DIR}/run.dev.small.tsv \
                                      --corpus ${DATA_DIR}/collection.tsv \
                                      --t5_input ${DATA_DIR}/query_doc_pairs.dev.small.txt \
                                      --t5_input_ids ${DATA_DIR}/query_doc_pair_ids.dev.small.tsv

split --suffix-length 3 --numeric-suffixes --lines 800000 ${DATA_DIR}/query_doc_pairs.dev.small.txt ${DATA_DIR}/query_doc_pairs.dev.small.txt

export GS_FOLDER=<google storage folder to store input/output data>
gsutil cp ${DATA_DIR}/query_doc_pairs.dev.small.txt??? ${GS_FOLDER}
