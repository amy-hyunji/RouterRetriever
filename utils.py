import os
import torch
import logging
import jsonlines
import numpy as np
import pandas as pd
import torch.distributed as dist

from tqdm import tqdm
from typing import Dict, List
from beir.util import cos_sim, dot_score
from beir.retrieval.search.base import BaseSearch
from beir.datasets.data_loader import GenericDataLoader
from collections import defaultdict

from src import normalize_text
import src.dist_utils as dist_utils 
from src.evaluate_retriever import GateEvaluateRetrieval
from src.exact_search import DenseRetrievalExactSearch 

logger = logging.getLogger(__name__)

def get_q_lora_path(gatename):
    raise NotImplementedError("Add path to LoRA ckpt of your own") 

def get_ctx_lora_path(gatename):
    raise NotImplementedError("Add path to LoRA ckpt of your own") 

### get query embeddings 
def get_qlist(args, dataname, split):
    qrel_path = os.path.join(args.beir_dir, dataname, "qrels", f"{split}.tsv")
    df = pd.read_csv(qrel_path, delimiter="\t")
    return list(set(list(df["query-id"])))

def get_queries(args, dataname, split):
    qid2query = {}
    qid_list = get_qlist(args, dataname, split)

    queries_path = os.path.join(args.beir_dir, dataname, "queries.jsonl")
    total_qid2query = {}
    with jsonlines.open(queries_path) as f:
        for elem in f.iter():
            total_qid2query[elem["_id"]] = elem["text"]
    for qid in qid_list:
        # qid = f"{dataname} :: {qid}"
        qid2query[qid] = total_qid2query[str(qid)] 
    assert len(qid_list) == len(qid2query), f"qid_list: {len(qid_list)}\nqid2query: {len(qid2query)}"
    return qid2query 

def get_corpus(args, dataname):
    cid2ctx = {}
    corpus_path = os.path.join(args.beir_dir, dataname, "corpus.jsonl")
    with jsonlines.open(corpus_path) as f:
        for elem in f.iter():
            cid2ctx[elem["_id"]] = elem
    return cid2ctx

def get_q2ctx(args, dataname, split):
    qrel_path = os.path.join(args.beir_dir, dataname, f"qrels/{split}.tsv")
    cid2ctx = get_corpus(args, dataname)
    df = pd.read_csv(qrel_path, delimiter="\t")
    qid2ctxs = defaultdict(list)
    for qid, cid, score in zip(df["query-id"], df["corpus-id"], df["score"]):
        qid = str(qid); cid = str(cid)
        if score == 1:
            qid = f"{dataname} :: {qid}"
            try:
                ctx = cid2ctx[cid]
                qid2ctxs[qid].append({"title": ctx["title"], "text": ctx["text"], "cid": cid, "dataset": dataname})
            except:
                continue
    return qid2ctxs

def get_ctx_text(elem):
    if elem["title"] == "":
        return elem["text"]
    else:
        return f"{elem['title']} : {elem['text']}"



def get_embeddings(id2sentences, base_model, tokenizer, batch_size=32):
    all_embeddings = []
    ids = list(id2sentences.keys())
    sentences = list(id2sentences.keys())
    cnt = len(sentences)

    for i in tqdm(range(0, cnt, batch_size)):
        batch_sentences = sentences[i:i + batch_size]
        
        # Apply tokenizer
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            outputs = base_model(**inputs)
        
        # Mean pooling
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
        
        embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
        all_embeddings.append(embeddings)

    id2emb = {}
    for _id, emb in zip(ids, all_embeddings):
        id2emb[_id] = emb 
    
    return id2emb 


def gate_evaluate_model(
    query_encoder,
    doc_encoder,
    tokenizer,
    dataset,
    batch_size,
    add_special_tokens=True,
    norm_query=False,
    norm_doc=False,
    is_main=True,
    split="test",
    score_function="dot",
    beir_dir="BEIR/datasets",
    lower_case=False,
    normalize_text=False,
    corpus_embs=None,
    corpus_ids=None,
    custom_queries=None,
):

    metrics = defaultdict(list)  # store final results

    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder

    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder,
            doc_encoder=doc_encoder,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            norm_query=norm_query,
            norm_doc=norm_doc,
            lower_case=lower_case,
            normalize_text=normalize_text,
        ),
        batch_size=batch_size,
    )
    retriever = GateEvaluateRetrieval(dmodel, score_function=score_function)

    dist_utils.barrier()

    data_path = os.path.join(beir_dir, dataset)

    if not os.path.isdir(data_path) and is_main:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = beir.util.download_and_unzip(url, beir_dir)
    print(f"Data path - {data_path}")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
    assert len(queries) == len(qrels)
    
    if custom_queries is not None:
        queries = custom_queries

    total_scores = {}
    if corpus_embs is None:
        corpus_embs, corpus_ids = retriever.retriever.get_corpus_emb(corpus)
    results = retriever.retrieve(corpus_embs, corpus_ids, corpus, queries)
    # print(results)
    # results = retriever.retrieve(corpus, queries)
    if is_main:
        ndcg, _map, recall, precision, scores = retriever.evaluate(qrels, results, retriever.k_values)
        for k, v in scores.items():
            assert k not in v
            total_scores[k] = v
        for metric in (ndcg, _map, recall):
            if isinstance(metric, str):
                metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
            for key, value in metric.items():
                metrics[key].append(value)

    metrics = {key: 100 * np.mean(value) for key, value in metrics.items()}

    return metrics, total_scores, corpus_ids, corpus_embs

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in tqdm(range(nbatch)):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [c["title"] + " " + c["text"] if len(c["title"]) > 0 else c["text"] for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.cuda() for key, value in cencode.items()}
                emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb
