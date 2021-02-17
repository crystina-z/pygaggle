"""
This script creates monoT5 input files for training,
Each line in the monoT5 input file follows the format:
    f'Query: {query} Document: {document} Relevant:\t{label}\n')
"""
import argparse
from tqdm import tqdm

def load_tsv(fn):
    print("loading tsv", fn)
    runs = {}
    with open(fn) as f:
        for line in f:
            id, txt = line.strip().split("\t")
            runs[id] = txt
    return runs


parser = argparse.ArgumentParser()
parser.add_argument(
        "--triples_train", "-t", type=str, required=True, help="tsv file <query>, <positive_document>, <negative_document>")
parser.add_argument(
        "--query_fn", type=str, default="/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/train_set/queries.train.tsv")
parser.add_argument(
        "--collection_fn", type=str, default="/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/collection.tsv")
parser.add_argument("--triples_type", type=str, default="id", choices=["txt", "id"])
parser.add_argument("--output_to_t5", "-o", type=str, required=True, help="t5 train input file")

args = parser.parse_args()

if args.triples_type == "id":
    id2query = load_tsv(args.query_fn)
    id2doc = load_tsv(args.collection_fn)
    print("loaded")

with open(args.output_to_t5, 'w') as fout_t5:
    for line_num, line in enumerate(tqdm(open(args.triples_train))):
        query, positive_document, negative_document = line.strip().split('\t')
        if args.triples_type == "id":
            query, positive_document, negative_document = id2query[query], id2doc[positive_document], id2doc[negative_document]

        fout_t5.write(f'Query: {query} Document: {positive_document} Relevant:\ttrue\n')
        fout_t5.write(f'Query: {query} Document: {negative_document} Relevant:\tfalse\n')

print('Done!')
