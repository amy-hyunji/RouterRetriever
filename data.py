import os
import sys
import copy
import json
import torch
import pickle
import random
import numpy as np

from src import normalize_text

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

class ContrieverDataset(Dataset):
    def __init__(
        self, 
        datapaths, 
        training,
        tokenizer,
        maxlength,
        negative_ctxs=1,
        dev_negative_ctxs=0,
        negative_hard_ratio=0.0,
        negative_hard_min_idx=0,
        normalize=False 
    ):
        super(ContrieverDataset, self).__init__()
        self.training=training
        
        assert datapaths.endswith('.json')
        self.dataset = json.load(open(datapaths))

        self.negative_ctxs = negative_ctxs 
        self.dev_negative_ctxs = dev_negative_ctxs 
        self.negative_hard_ratio = negative_hard_ratio 
        self.negative_hard_min_idx = negative_hard_min_idx 
        self.normalize_fn = normalize_text.normalize if normalize_text else lambda x: x 

        self.tokenizer = tokenizer
        self.passage_maxlength = maxlength

    def sample_n_hard_negatives(self, ex):

        if "hard_negative_ctxs" in ex and len(ex["hard_negative_ctxs"]) > 0:
            n_hard_negatives = sum([random.random() < self.negative_hard_ratio for _ in range(self.negative_ctxs)])
            n_hard_negatives = min(n_hard_negatives, len(ex["hard_negative_ctxs"][self.negative_hard_min_idx :]))
        else:
            n_hard_negatives = 0

        n_random_negatives = self.negative_ctxs - n_hard_negatives

        if "negative_ctxs" in ex and len(ex["negative_ctxs"]) > 0:
            n_random_negatives = min(n_random_negatives, len(ex["negative_ctxs"]))
        else:
            n_random_negatives = 0
        return n_hard_negatives, n_random_negatives

    def __len__(self):
        return len(self.dataset) 

    def __getitem__(self, idx):
        example = self.dataset[idx]

        question = example["question"]

        if self.training:
            gold = random.choice(example["positive_ctxs"])

            n_hard_negatives, n_random_negatives = self.sample_n_hard_negatives(example)
            negatives = []

            if n_random_negatives > 0:
                random_negatives = random.sample(example["negative_ctxs"], n_random_negatives)
                negatives += random_negatives 
            if n_hard_negatives > 0:
                hard_negatives = random.sample(
                    example["hard_negative_ctxs"][self.negative_hard_min_idx:], n_hard_negatives
                )
                negatives += hard_negatives

        else:
            gold = example["positive_ctxs"][0]
            nidx = 0
            if self.dev_negative_ctxs > 0 and "negative_ctxs" in example:
                negatives = [example["negative_ctxs"][nidx]]
            else:
                negatives = []

        gold = gold["title"] + " " + gold["text"] if "title" in gold and len(gold["title"]) > 0 else gold["text"] 

        negatives = [
            n["title"] + " " + n["text"] if ("title" in n and len(n["title"]) > 0) else n["text"] for n in negatives
        ]

        example = {
            "query": self.normalize_fn(question),
            "gold": self.normalize_fn(gold),
            "negatives": [self.normalize_fn(n) for n in negatives] 
        }

        queries = [example["query"]]
        golds = [example["gold"]]
        negs = example["negatives"]
        #assert len(negs) > 0
        allpassages = golds + negs

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )
        kout = self.tokenizer.batch_encode_plus(
            allpassages,
            max_length=self.passage_maxlength,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )

        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(golds)], k_mask[: len(golds)]
        if len(golds) == len(k_tokens):
            n_tokens=n_mask=[]
        else:
            n_tokens, n_mask = k_tokens[len(golds) :].squeeze(), k_mask[len(golds) :].squeeze()

        batch = {
            "q_tokens": q_tokens.squeeze(),
            "q_mask": q_mask.squeeze(),
            "k_tokens": k_tokens.squeeze(),
            "k_mask": k_mask.squeeze(),
            "g_tokens": g_tokens.squeeze(),
            "g_mask": g_mask.squeeze(),
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        
        return batch
