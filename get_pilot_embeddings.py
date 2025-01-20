import os
import json
import numpy 
import torch
import random
import argparse
import jsonlines

import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import defaultdict
from transformers import AutoTokenizer

from src.contriever import Contriever
from utils import get_queries, get_ctx_text, get_corpus, get_embeddings



def get_all(args, dataname, base_model, tokenizer, get_ctx=False):
    print(f"Dumping Embedding for {dataname}")
    split = "train" 
    qid2query = get_queries(args, dataname, split)
    if args.sample_data is not None:
        items = list(qid2query.items())
        random.shuffle(items)
        qid2query = dict(items[:args.sample_data])
    qid2emb = get_embeddings(qid2query, base_model, tokenizer, batch_size=args.per_gpu_batch_size)
    
    if get_ctx: 
        ctx_list = []
        cid2ctx = get_corpus(args, dataname)
        for ctx in cid2ctx.values(): 
            ctx_list.append(get_ctx_text(ctx))
        print(f"Example of {dataname} : {ctx_list[0]}")
        if args.sample_data is not None:
            random.shuffle(ctx_list)
            ctx_list = ctx_list[:args.sample_data]
        cid2emb = get_embeddings(ctx_list, batch_size=args.per_gpu_batch_size)
    else:
        cid2emb = None
    return qid2emb, cid2emb 

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    base_model = Contriever.from_pretrained(args.base_encoder)
    return tokenizer, base_model 

def do_grouping(score_dict):
    groups = defaultdict(list) 
    for qid, gatename2scoredict in score_dict.items():
        score_dict = {gname: score["ndcg@10"] for gname, score in gatename2scoredict.items()}
        max_data = max(score_dict, key=score_dict.get)
        groups[max_data].append(qid)
    return groups

def get_pilot_emb(args, dataset_list, base_model, tokenizer):
    pilot_embs = defaultdict(list)
    for dataset in dataset_list:
        print(f"Constructing pilot embeddings for .. {dataset}")
        score_path = os.path.join(args.score_path, f"data_{dataset}.all_score.json")
        score_dict = json.load(open(score_path))
        dataset_group = do_grouping(score_dict)
        
        ## get embeddings of the dataset - {qid: emb}
        qid2emb, _ = get_all(args, dataset, base_model, tokenizer)

        ## do clustering
        for max_data, qidlist in dataset_group.items():
            emblist = [qid2emb[_id] for _id in qidlist] 
            kmeans = KMeans(n_clusters=args.k, random_state=args.random_state)
            clusters = kmeans.fit_predict(emblist.numpy()) 
            for emb in kmeans.cluster_centers_: # centroid embs - shape (k, d)
                pilot_embs[max_data].append(emb)

    return pilot_embs

def main(args):

    tokenizer, base_model = load_model(args)
    dataset_list = args.datasets.split("/")
    pilot_embs = get_pilot_emb(args, dataset_list, base_model, tokenizer)
    print(f"Saving pilot embeddings to .. {args.save_path}")
    with open(args.save_path, "w") as f:
        json.dump(pilot_embs, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=None, help="name of datasets to construct pilot embeddings. separated by `/`")
    parser.add_argument("--base_encoder", type=str, default="facebook/contriever")
    parser.add_argument("--sample_data", type=int, default=None)
    parser.add_argument("--per_gpu_batch_size", type=int, default=128)
    parser.add_argument("--score_path", type=str, default=None, help="path to score dict")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--save_path", type=str, default="./", help="path to save pilot embeddings")
    parser.add_argument("--k", type=int, default=1, help="K value for KMeans Clustering")
    parser.add_argument("--random_state", type=int, default=42, help="random state for KMeans clustering")
    args = parser.parse_args()
    main(args)