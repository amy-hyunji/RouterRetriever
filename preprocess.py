import os
import csv
import json
import argparse
import jsonlines

def convert(args, dataname):
    basepath = os.path.join(args.beir_dir, dataname) 

    corpus = "corpus.jsonl"
    queries = "queries.jsonl"
    qrels = "qrels/train.tsv"

    save_list = []
    with open(os.path.join(basepath, qrels), mode="r", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        data = [row for row in reader]

    qid2qtext = {}
    cid2ctext = {}

    with jsonlines.open(os.path.join(basepath, queries)) as f:
        for elem in f:
            _id = elem["_id"]
            text = elem["text"]
            qid2qtext[_id] = text

    with jsonlines.open(os.path.join(basepath, corpus)) as f:
        for elem in f:
            _id = elem["_id"]
            title = elem["title"]
            text = elem["text"]
            ctxs = {"title": title, "text": text}
            cid2ctext[_id] = ctxs

    for elem in data:
        qid = elem["query-id"]
        cid = elem["corpus-id"]
        save_list.append({"question": qid2qtext[qid], "positive_ctxs": [cid2ctext[cid]]})

    savepath = os.path.join(args.save_dir, dataname) 
    os.makedirs(savepath, exist_ok=True)

    with open(os.path.join(savepath, "train.json") ,"w") as f:
        json.dump(save_list, f)

    print(f"Done converting and saving {dataname} to {savepath}")

def main(args):
    datalist = args.datasets.split("/")
    for dataname in datalist:
        convert(args, dataname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--datasets", type=str, help="names of evaluation dataset from the BEIR benchmark to convert. separated by `/`")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")        
    parser.add_argument("--save_dir", type=str, default="./beir_datasets", help="path to save score")

    args, _ = parser.parse_known_args()
    main(args)