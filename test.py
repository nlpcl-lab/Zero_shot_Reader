import argparse, time, json, os
import logging
from beir import LoggingHandler
import pytorch_lightning as pl
from src.toy import QASDataset, Reader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar

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
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")

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

    data_path = "./data/nq/nq-dev.json"
    dataloader = model.get_dataloader(data_path)

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=CSVLogger(save_dir="./logs/"), callbacks=[TQDMProgressBar(refresh_rate=10)])
    trainer.test(model, dataloaders=dataloader)


if __name__=="__main__":
    main()
