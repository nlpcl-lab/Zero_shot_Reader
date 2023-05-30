"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch.

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from IPython import embed

import argparse
import pathlib, os, random, json
import logging
from IPython import embed

def parse():
    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--dataset", type=str, default="trivia")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--dataset_dir", type=str, default="./data/")
    parser.add_argument("--retriever", type=str, default="dpr")
    args = parser.parse_args()

    return args

def load_data(dataset, data_path, split):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    data_dir = os.path.join(data_path, dataset)
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset

    #### Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels


#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/
def bm25(corpus, queries, dataset):
    hostname = "localhost" #localhost
    index_name = "nq" # scifact

    #### Intialize ####
    # (1) True - Delete existing index and re-index all documents from scratch
    # (2) False - Load existing index
    initialize = False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
    # SciFact is a relatively small dataset! (limit shards to 1)
    number_of_shards = 1
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    # (2) For datasets with big corpus ==> keep default configuration
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model, k_values=[1,3,5,10,100,1000])

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)


    with open("{}-bm25.json".format(dataset), 'w') as f:
        json.dump(results, f)
    return retriever, results

def dpr(corpus, queries, dataset):
    model = DRES(models.SentenceBERT((
        "facebook-dpr-question_encoder-multiset-base",
        "facebook-dpr-ctx_encoder-multiset-base",
        " [SEP] "), batch_size=128))
    retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1,3,5,10,100,1000])
    results = retriever.retrieve(corpus, queries)

    # retriever = EvaluateRetrieval()
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    # ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 100, 1000])
    # top_k = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100,1000], metric="top_k_acc")
    # embed()

    for k,v in results.items():
        v = {a:b for a,b in sorted(v.items(),key=lambda item: item[1], reverse=True)[:1000]}
        temp = {}
        for a,b in v.items():
            temp[a] = b
        results[k] = temp

    with open("{}-dpr.json".format(dataset), 'w') as f:
        json.dump(results, f)
    return retriever, results


def evaluate(retriever, results, qrels, corpus):
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))

    #### Retrieval Example ####
    query_id, scores_dict = random.choice(list(results.items()))
    logging.info("Query : %s\n" % queries[query_id])

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))


if __name__=="__main__":
    args = parse()
    corpus, queries, qrels = load_data(args.dataset, args.dataset_dir, args.split)

    if args.retriever == "bm25":
        retriever, results = bm25(corpus, queries, args.dataset)
    elif args.retriever == "dpr":
        retriever, results = dpr(corpus, queries, args.dataset)