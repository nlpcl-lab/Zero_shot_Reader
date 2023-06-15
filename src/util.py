import os, re, string, json
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, Sequence, Optional
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple
from tqdm.autonotebook import tqdm
import json
import os
import logging
import csv
from sklearn.utils.extmath import softmax
import numpy as np
from IPython import embed


logger = logging.getLogger(__name__)

class GenericDataLoader:

    def __init__(self, data_folder: str = None, prefix: str = None, corpus_file: str = "corpus.jsonl",
                 query_file: str = "queries.jsonl",
                 qrels_folder: str = "qrels", qrels_file: str = ""):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(data_folder, corpus_file) if data_folder else corpus_file
        self.query_file = os.path.join(data_folder, query_file) if data_folder else query_file
        self.qrels_folder = os.path.join(data_folder, qrels_folder) if data_folder else None
        self.qrels_file = qrels_file

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError("File {} not present! Please provide accurate file.".format(fIn))

        if not fIn.endswith(ext):
            raise ValueError("File {} must be present with extension {}".format(fIn, ext))

    def load_custom(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load(self, split="test") -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="jsonl")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self._load_queries()

        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels

    def load_corpus(self) -> Dict[str, Dict[str, str]]:

        self.check(fIn=self.corpus_file, ext="jsonl")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus

    def _load_corpus(self):

        num_lines = sum(1 for i in open(self.corpus_file, 'rb'))
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }

    def _load_queries(self):

        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = {"text": line.get("text"), "answer": line.get("answer")}

    def _load_qrels(self):

        reader = csv.reader(open(self.qrels_file, encoding="utf-8"),
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])

            if query_id not in self.qrels:
                self.qrels[query_id] = {corpus_id: score}
            else:
                self.qrels[query_id][corpus_id] = score


def load_data(dataset, dataset_dir, split):
    if os.path.exists(os.path.join(dataset_dir, dataset)):
        data_path = os.path.join(dataset_dir, dataset)
    else:
        exit(0)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _accuracy(golds, pred):
    cor = max([gold == pred for gold in golds])
    return cor

def get_result(predictions):
    pos_result = []
    neg_result = []
    for acc,score,_,_ in predictions[0]:
        if acc > 0:
            pos_result.append(score)
        else:
            if score > 0:
                neg_result.append(score)

    return np.array(pos_result), np.array(neg_result)

def get_result2(predictions):
    pos_result = []
    neg_result = []
    for acc,score,_,_ in predictions:
        if acc > 0:
            pos_result.append(score)
        else:
            if score > 0:
                neg_result.append(score)

    return np.array(pos_result), np.array(neg_result)

def _calculate_score(final_scores, preds,answers, UC):
    if UC:
        preds = np.array(preds)
        final_scores = np.array(final_scores)
        index = preds != "unanswerable"
        preds = preds[index]
        if len(preds) == 0:
            return 0
        else:
            final_scores = final_scores[index]
            score = max(final_scores)
            pred = preds[np.where(final_scores == score)[0][0]]
    else:
        score = max(final_scores)
        pred = preds[final_scores.index(score)]

    return _accuracy(answers, pred)


def calculate_score(out_dir, raw_result, UC):
    top_k = [10,20,30,40,50,60,70,80,90,100]
    nc_scores = {}
    scores = {}
    for i in top_k:
        scores[i] = 0
        nc_scores[i] = 0
        for _,v in raw_result.items():
            # answers = v['ids']
            answers = v['answers']
            preds = v['preds'][:i]
            answer_scores = v['as'][:i]
            rel_scores = softmax(np.array([v['rs'][:i]])).tolist()[0]
            # Not NC
            scores[i] += _calculate_score(answer_scores,preds,answers,UC)
            final_scores = [s*r for s,r in zip(answer_scores, rel_scores)]
            nc_scores[i] += _calculate_score(final_scores,preds,answers,UC)
        scores[i] = scores[i]/len(raw_result)
        nc_scores[i] = nc_scores[i]/len(raw_result)
    print(scores)
    print(nc_scores)
    with open(os.path.join(out_dir, 'scores.json'),'w') as f:
        json.dump(scores, f)

    with open(os.path.join(out_dir, 'nc_scores.json'),'w') as f:
        json.dump(nc_scores, f)


class CustomWriter(BasePredictionWriter):

    def __init__(self, out_dir, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.results = []
        self.out_dir = out_dir
        self.logger = logging.getLogger(type(self).__name__)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.results = [[] for _ in range(trainer.world_size)]

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        pos_result, neg_result = get_result(predictions)
        plt.hist(pos_result, bins=100, alpha=0.5)
        plt.hist(neg_result, bins=100, alpha=0.5)
        plt.title("Acc {}".format(round(len(pos_result)/len(predictions[0])*100,2)))

        plt.savefig(self.out_dir)

class CustomWriter2(BasePredictionWriter):

    def __init__(self, out_dir, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.results = []
        self.out_dir = out_dir
        self.logger = logging.getLogger(type(self).__name__)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.results = [[] for _ in range(trainer.world_size)]

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        raw_result = {}
        for r in predictions[0]:
            raw_result.update(r)
        with open(os.path.join(self.out_dir, "raw_result.json"), 'w') as f:
            json.dump(raw_result, f)


class CustomWriter3(BasePredictionWriter):

    def __init__(self, out_dir, wrtie_interval="epoch"):
        super().__init__(wrtie_interval)
        self.results = []
        self.out_dir = out_dir
        self.logger = logging.getLogger(type(self).__name__)

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        self.results = [[] for _ in range(trainer.world_size)]

    def write_on_epoch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]],
    ):
        raw_result = {}
        for  r in predictions:
            raw_result.update(r)
        with open(os.path.join(self.out_dir, "raw_result.json"), 'w') as f:
            json.dump(raw_result, f)

