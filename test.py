import argparse, time, json, os
import logging
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
import pytorch_lightning as pl
from src.toy import Reader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from src.util import CustomWriter, load_data
from IPython import embed

def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

def parse():
    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--dataset_dir", type=str, default="./data/")
    parser.add_argument("--batch", type=int, default=1)

    #Reader
    parser.add_argument("--model", type=str, default="opt-iml-1.3b")
    parser.add_argument("--cs", type=int, default=0)
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--uncertain", type=int, default=0)

    args = parser.parse_args()

    return args

def main():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    t = timestr()
    args = parse()
    logging.info("Implemented time is {}".format(t))
    model = Reader(args)

    corpus, queries, qrels = load_data('nq', "./data", 'dev')
    with open('./data/nq/nq-dev-DPR.json') as f:
        results = json.load(f)

    retriever = EvaluateRetrieval()
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 100, 1000])
    top_k = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100,1000], metric="top_k_acc")
    # data_path = "./data/nq/nq-dev.json"
    # dataloader = model.get_dataloader(data_path)
    dataloader = model.get_dataloader(corpus, queries, results)

    writer = CustomWriter("./output/reader_{}_retriever_dpr_docs_{}_uncertain_{}.jpg".format(args.model.split("/")[-1], args.num_docs, args.uncertain))
    trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader)


if __name__=="__main__":
    main()
