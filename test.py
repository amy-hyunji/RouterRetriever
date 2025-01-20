import json
import argparse
import numpy as np

from peft import PeftModel
from collections import defaultdict
from transformers import AutoTokenizer
from beir.utils import cos_sim, dot_score

from src.contriever import Contriever
from utils import get_lora_path, gate_evaluate_model, get_embeddings, get_queries 

## Load pilot embs
def load_pilot_embs(args):
    return json.load(open(args.pilot_embs_path)) 

def calculate_similarity(score_function, emb1, emb2):
    if score_function == "dot":
        return dot_score(emb1, emb2)
    elif score_function == "cos_sim":
        return cos_sim(emb1, emb2)
    else:
        raise NotImplementedError(f"Choose score function between (dot|cos_sim)") 

## calculate similarity score with pilot embs and find the expert to pass through 
def find_expert(args, dataname, tokenizer, base_model, pilot_embs):

    split = "dev" if dataname == "msmarco" else "test"
    qid2query = get_queries(args, dataname, split)

    # get query_emb 
    qid2embs = get_embeddings(qid2query, base_model, tokenizer, args.per_gpu_batch_size) 

    # route and find the expert
    all_pilot_embs = []
    expert_indices = []

    for expert, embeddings in pilot_embs.items():
        all_pilot_embs.extend(embeddings)
        expert_indices.extend([expert] * len(embeddings))

    all_pilot_embs = np.array(all_pilot_embs)

    expert2qid = defaultdict(list) 
    for qid, query_emb in qid2embs.items():
        similarities = [calculate_similarity(args.score_function, query_emb, pilot_emb) for pilot_emb in all_pilot_embs]
        best_index = np.argmax(similarities)
        best_expert = expert_indices[best_index]
        expert2qid[best_expert].append(qid)

    return expert2qid, qid2query

def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_encoder)
    base_model = Contriever.from_pretrained(args.base_encoder)
    return tokenizer, base_model 

def load_q_encoder(expert):
    query_encoder = Contriever.from_pretrained(args.base_encoder)
    lora_model = get_lora_path(expert)
    query_encoder = PeftModel.from_pretrained(query_encoder, lora_model, adapter_name=expert)
    query_encoder.set_adapter(adapter_name=expert)
    query_encoder.cuda()
    return query_encoder

def main(args):
    tokenizer, doc_encoder = load_model(args)
    doc_encoder.cuda()
    dataset_list = args.datasets.split("/")
    pilot_embs = load_pilot_embs(args) # {expert name: [list of pilot embs]}

    score_dict = defaultdict(list)  # score_dict = {dataname: [list of scores]} 

    for dataset in dataset_list:
        
        # route through expert
        expert2qid, qid2query = find_expert(args, dataset, tokenizer, doc_encoder, pilot_embs)

        for expert, qidlist in expert2qid.items():

            # construct custom_queries 
            custom_queries = {qid: qid2query[qid] for qid in qidlist}
            query_encoder = load_q_encoder(expert)
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
                custom_queries = custom_queries 
            )

            for v in scores.values():
                score_dict[dataset].append(v['ndcg_cut_10'])

    print(f"Saving result in {args.save_dir}")
    with open(args.save_dir, "w") as f:
        json.dump(score_dict, f) 

    print("*"*80)
    for data, scorelist in score_dict.items():
        avg_score = np.array(scorelist).mean()
        print(f"[{data}]] - {avg_score}")
    print("*"*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--base_encoder", type=str, default="facebook/contriever", help="Model name or path")
    parser.add_argument("--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings")
    parser.add_argument("--pilot_embs_path", type=str, default="./", help="path to pilot embeddings")
    parser.add_argument("--datasets", type=str, help="names of evaluation dataset from the BEIR benchmark. separated by `/`")
    parser.add_argument("--beir_dir", type=str, default="./", help="Directory to save and load beir datasets")        
    parser.add_argument("--per_gpu_batch_size", type=int, default=128)
    parser.add_argument(
        "--score_function", type=str, default="dot", help="Metric used to compute similarity between two embeddings"
    )
    parser.add_argument("--save_dir", type=str, default="score.json", help="path to save score")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")
    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")
    parser.add_argument("--lower_case", action="store_true", help="lowercase query and document text")
    parser.add_argument("--normalize_text", action="store_true", help="Apply function to normalize some common characters")

    args, _ = parser.parse_known_args()
    main(args)
