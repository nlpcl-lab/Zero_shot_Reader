import argparse, time, json, os
import logging
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
import pytorch_lightning as pl
from src.model import Reader
from src.util import load_data, _normalize_answer
from IPython import embed

def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

def parse():
    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--retriever", type=str, default="DPR")

    #Path
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./output")

    #Output

    #Default
    # parser.add_argument("--gpu", type=str, default="gpu10")
    # parser.add_argument("--date", type=str, default="20230514-191652")
    #NC
    # parser.add_argument("--gpu", type=str, default="A100/1")
    # parser.add_argument("--date", type=str, default="20230601-184144")
    #UC
    # parser.add_argument("--gpu", type=str, default="gpu11")
    # parser.add_argument("--date", type=str, default="20230522-164739")

    #UC+NC
    parser.add_argument("--gpu", type=str, default="A100/2")
    parser.add_argument("--date", type=str, default="20230606-115100")


    args = parser.parse_args()

    return args

def _accuracy(golds, pred):
    cor = max([gold == pred for gold in golds])
    return cor

def main():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    args = parse()

    corpus, queries, qrels = load_data(args.dataset, args.dataset_dir, args.split)
    with open(os.path.join(args.output_dir, args.gpu, args.date, "raw_result.json")) as f:
        results = json.load(f)

    analysis = [0,0,0,0,0]
    # for q in results.keys():
    #     rel_context = list(qrels[q].keys())
    #     top_1_answer = sorted(results[q].items(), key=lambda item: item[1]['score'], reverse=True)[0]
    #     context, answer = top_1_answer[0], top_1_answer[1]['pred']
    #     nor_answer = _normalize_answer(answer)
    #     targets = [_normalize_answer(a) for a in queries[q]["answer"]]
    #     if context in rel_context:
    #         if _accuracy(targets, nor_answer):
    #             analysis[0] += 1
    #         else:
    #             analysis[1] += 1
    #     else:
    #         if _accuracy(targets, nor_answer):
    #             analysis[2] += 1
    #         else:
    #             analysis[3] += 1

    for q in results.keys():
        rel_context = list(qrels[q].keys())
        filter_results = {}
        for d,v in results[q].items():
            if v["pred"] != "unanswerable":
                filter_results[d] = v
        if len(filter_results) == 0:
            analysis[4] += 1
        else:
            top_1_answer = sorted(filter_results.items(), key=lambda item: item[1]['score'], reverse=True)[0]
            context, answer = top_1_answer[0], top_1_answer[1]['pred']
            nor_answer = _normalize_answer(answer)
            targets = [_normalize_answer(a) for a in queries[q]["answer"]]
            if context in rel_context:
                if _accuracy(targets, nor_answer):
                    analysis[0] += 1
                else:
                    analysis[1] += 1
            else:
                if _accuracy(targets, nor_answer):
                    analysis[2] += 1
                else:
                    analysis[3] += 1
    embed(); exit(0)


if __name__=="__main__":
    main()
