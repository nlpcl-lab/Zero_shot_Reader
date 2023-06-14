import argparse, time, json, os
import logging
from beir import LoggingHandler
from beir.retrieval.evaluation import EvaluateRetrieval
import pytorch_lightning as pl
from src.model import Reader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from src.util import CustomWriter2, CustomWriter3, load_data, calculate_score
from IPython import embed

def timestr():
    return time.strftime("%Y%m%d-%H%M%S")

def parse():
    parser = argparse.ArgumentParser()

    #Dataset
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--retriever", type=str, default="DPR")

    #Dataloader
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=10)

    #Reader
    parser.add_argument("--model", type=str, default="opt-iml-1.3b")
    parser.add_argument("--num_docs", type=int, default=100)

    #Prompt Type
    parser.add_argument("--prompt", type=str, default="Read the following context and answer the question.")
    parser.add_argument("--CoT", action="store_true")
    parser.add_argument("--SC", action="store_true")
    parser.add_argument("--UC", action="store_true")

    #Output Verbalizer
    parser.add_argument("--output_verbalizer", type=str, default="Answer:")

    #Noisy Channel
    parser.add_argument("--NC", action="store_true")

    #Path
    parser.add_argument("--dataset_dir", type=str, default="./data")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--output_dir", type=str, default="./output")


    #ETC
    parser.add_argument("--a100", action="store_true")
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

    corpus, queries, qrels = load_data(args.dataset, args.dataset_dir, args.split)

    with open(os.path.join(args.dataset_dir,'{d}/{d}-{s}-{r}.json'.format(d=args.dataset, s=args.split, r=args.retriever))) as f:
        results = json.load(f)

    retriever = EvaluateRetrieval()
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    retriever.evaluate(qrels, results, [1, 3, 5, 10, 20, 100, 1000])
    retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100,1000], metric="top_k_acc")


    model = Reader(args)
    dataloader = model.get_dataloader(corpus, queries, results, args.num_workers)

    out_dir = os.path.join(args.output_dir,t)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f)

    if args.a100:
        writer = CustomWriter3(out_dir)
    else:
        writer = CustomWriter2(out_dir)
    trainer = pl.Trainer(accelerator="gpu", devices=1, callbacks=writer)
    trainer.predict(model, dataloaders=dataloader)

    with open(os.path.join(out_dir, 'raw_result.json')) as f:
        result = json.load(f)
    calculate_score(out_dir, result, args.UC)


if __name__=="__main__":
    main()
