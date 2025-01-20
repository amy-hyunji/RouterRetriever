# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import argparse
import torch
import logging
import json
import numpy as np
import os
import copy

import src.gate_contriever
import src.beir_utils
import src.utils
import src.dist_utils
import src.contriever
from src.contriever import Contriever

import src.pt_contriever

from collections import defaultdict
from peft import PeftModel
from transformers import AutoTokenizer


from utils import get_lora_path, gate_evaluate_model 

logger = logging.getLogger(__name__)

def main(args):

    logger = src.utils.init_logger(args)

    doc_encoder = Contriever.from_pretrained(args.base_encoder)
    doc_encoder.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    
    logger.info("Start indexing")
    assert args.save_dir != ""
    
    path = os.path.join(args.save_dir, f"data_{args.dataset}.all_score.json")
    tmp_path = os.path.join(args.save_dir, f"data_{args.dataset}.all_score.tmp")
    gatelist = args.gatenames.split("/")
    logger.info(f"Running over gates: {gatelist}")
    
    if os.path.exists(path):
        logger.info("Done dumping file!") 
        q2scores = json.load(open(path))
        qidlist = list(q2scores.keys())
        exist_gate = list(q2scores[qidlist[0]].keys()) 
        n_gate = []
        for gatename in gatelist:
            if gatename not in exist_gate:
                n_gate.append(gatename)
        if len(n_gate) == 0:
            logger.info(f"Done dumping score for {exist_gate}")
        else:
            gatelist = n_gate # evaluate over only the missing gates!
    elif os.path.exists(tmp_path): 
        logger.info("Resuming from existing file!") 
        q2scores = json.load(open(tmp_path))
        qidlist = list(q2scores.keys())
        exist_gate = list(q2scores[qidlist[0]].keys())
        assert exist_gate == gatelist, f"When resuming from tmp file, please keep only the existing gates: {exist_gate}. You are requesting: {gatelist}"
    else:
        q2scores = defaultdict(dict)
        exist_gate = []

    for gatename in gatelist:
        if gatename in exist_gate:
            logger.info(f"Passing score with gate: {gatename} | Exist: {exist_gate}")
            continue
        else:
            logger.info(f"Dumping score with gate: {gatename}")
            
            # Loading Query
            query_encoder = Contriever.from_pretrained(args.base_encoder)
            lora_model = get_lora_path(gatename)
            query_encoder = PeftModel.from_pretrained(query_encoder, lora_model, adapter_name=gatename)
            query_encoder.set_adapter(adapter_name=gatename)
            query_encoder.cuda()

            _, scores, _, _ = gate_evaluate_model(
                query_encoder=query_encoder,
                doc_encoder=doc_encoder,
                tokenizer=tokenizer,
                dataset=args.dataset,
                batch_size=args.per_gpu_batch_size,
                norm_query=args.norm_query,
                norm_doc=args.norm_doc,
                is_main=src.dist_utils.is_main(),
                split="dev" if args.dataset == "msmarco" else "test",
                score_function=args.score_function,
                beir_dir=args.beir_dir,
                lower_case=args.lower_case,
                normalize_text=args.normalize_text,
            )

            for query_id, v in scores.items():
                q2scores[query_id][gatename] = {"ndcg@10": v['ndcg_cut_10'], "recall@10": v["recall_10"], "recall@100": v["recall_100"]} 

        os.makedirs(args.save_dir, exist_ok=True) 
        logger.info(f"Save in .. {tmp_path}")
        with open(tmp_path, "w") as f:
            json.dump(q2scores, f)
    
    with open(path, "w") as f:
        json.dump(q2scores, f)
    os.system(f"rm {tmp_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--base_encoder", type=str, default="facebook/contriever", help="Model name or path")
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument(
        "--normalize_text", action="store_true", help="Apply function to normalize some common characters"
    )
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--gatenames", type=str, default="", help="names of gates to get score separated by `/`")

    args, _ = parser.parse_known_args()
    main(args)
