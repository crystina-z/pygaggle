import os
from tqdm import tqdm
from argparse import ArgumentParser 
from collections import defaultdict
from nirtools.ir import load_qrels


def get_args():
    parser = ArgumentParser() 
    # parser.add_argument("--resmapling_type", "-t", )
    parser.add_argument("--output_fn", "-o", required=True)
    return parser.parse_args()


def get_file_line_number(fn):
    return int(os.popen(f"wc -l {fn}").readline().split()[0])


def read_file(fn):
    with open(fn) as f:
        for line in tqdm(f, desc=f"reading {fn}", total=get_file_line_number(fn)):
            yield line


def load_qids(fn):
    qids = set()
    for line in read_file(fn):
        qid = line.strip().split()[0]
        qids.add(qid)
    return qids
 

def load_bm25_in_batch(fn, qrels):
    all_qids = list(load_qids(fn))
    n = 1
    stride = len(all_qids) // n
    for i in range(0, len(all_qids), stride):
        print(f"the {i}-th batch, {i}")
        runs = defaultdict(list)
        qids = all_qids[i:i+stride]
        for line in read_file(fn):
            # jimport pdb
            # pdb.set_trace()
            qid, doc = line.strip().split()
            if qid not in qids:
                continue
            if qrels[qid].get(doc, 0) == 1:
                continue
            runs[qid].append(doc)
        yield runs
 

# in id format
origin_triple_fn = "/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/triples.train.small.idversion.tsv"
bm25_fn = "/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/top1000.train.ids"
qrels_fn = "/home/crystina/src/sigir2021/pygaggle/data/msmarco_passage/qrels.train.tsv"
qrels = load_qrels(qrels_fn)
print("loaded qrels")
 
args = get_args()
fout = open(args.output_fn, "w") 
for negruns in load_bm25_in_batch(bm25_fn, qrels):
    with open(origin_triple_fn) as f: 
        for line in tqdm(f):
            qid, pos, _ = line.strip().split()
            if qid not in negruns:
                continue
            neg = random.choice(negruns[qid])
            fout.write(f"{qid}\t{pos}\t{neg}\n")
    fout.flush()
fout.close()
print("finished")

