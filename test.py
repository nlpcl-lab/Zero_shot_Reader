import argparse, time, json, os
import logging
from beir import LoggingHandler
import pytorch_lightning as pl
from src.toy import QASDataset, Reader, Phrase_QASDataset, Phrase_Reader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from src.util import CustomWriter, load_data

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
    parser.add_argument("--model", type=str, default="facebook/opt-2.7b")
    parser.add_argument("--cs", type=int, default=0)
    parser.add_argument("--num_docs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)

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

    corpus, queries, _ = load_data('nq', "./data", 'dev')
    with open('./data/nq/nq-dev-bm25.json') as f:
        qrels = json.load(f)
    # data_path = "./data/nq/nq-dev.json"
    # dataloader = model.get_dataloader(data_path)
    dataloader = model.get_dataloader(corpus, queries, qrels)

    writer = CustomWriter("./output/BM25_{}_{}_{}.jpg".format(args.num_docs, args.cs, args.threshold))
    trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader)


if __name__=="__main__":
    main()
