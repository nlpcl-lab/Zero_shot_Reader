from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

from transformers import DPRReader, DPRReaderTokenizer
from torch.utils.data import DataLoader, Dataset
from src.util import _normalize_answer
import pytorch_lightning as pl
import json
from spacy.lang.en import English
from torch.nn.functional import log_softmax
from torch import topk
from tqdm import tqdm
from math import exp
import os
import torch
from IPython import embed
import numpy as np

def get_model_from_huggingface(model_dir, model):
    model_path = os.path.join(model_dir, model)
    if "opt" in model:
        return AutoModelForCausalLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)
    if "t5" in model:
        return AutoModelForSeq2SeqLM.from_pretrained(model_path), AutoTokenizer.from_pretrained(model_path)
    if "T0" in model:
        return T5ForConditionalGeneration.from_pretrained(model_path), T5Tokenizer.from_pretrained(model_path)
    return None


class QASDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.context, self.question, self.answer = [], [], []
        with open(file_path, 'r') as f:
            data = json.load(f)
        for d in data:
            self.question.append(d['question'])
            if len(d['positive_ctxs']) >=3:
                self.context.append((d['positive_ctxs'][0]['text'], d['positive_ctxs'][1]['text'], d['positive_ctxs'][2]['text']))
            elif len(d['positive_ctxs']) >=2:
                self.context.append((d['positive_ctxs'][0]['text'], d['positive_ctxs'][1]['text']))
            else:
                self.context.append((d['positive_ctxs'][0]['text']))
            self.answer.append(d['answers'])


    def __len__(self):
        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answer)
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.question[i], self.answer[i]

class Retrieved_Dataset(Dataset):
    def __init__(self, corpus, queries, qrels, num_docs):
        super().__init__()
        self.context, self.question, self.answer, self.ids = [], [], [], []
        for q,documents in qrels.items():
            self.question.append(queries[q]['text'])
            self.answer.append(queries[q]['answer'])
            self.context.append([corpus[d]['text'] for d in documents.keys()][:num_docs])
            self.ids.append({q:[d for d in list(documents.keys())[:num_docs]]})


    def __len__(self):
        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answer)
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.question[i], self.answer[i], self.ids[i]

class Reader(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model
        self.model, self.tokenizer = get_model_from_huggingface(args.model_dir, args.model)
        if "opt" in self.model_name:
            self.tokenizer.padding_side = "left"

        self.template = "{p}\n\nContext: {d}\nQuestion: {q}\nAnswer:"
        self.prompt = args.prompt
        if args.CoT:
            self.prompt = self.prompt.replace(".","") + " with reasoning step-by-step."
        if args.SC:
            pass #TO-DO
        if args.UC:
            self.prompt = self.prompt + ' If you can\'t find an answer, return "unanswerable".'

        self.batch_size = args.batch
        self.CoT = args.CoT
        self.SC = args.SC
        self.UC = args.UC
        self.NC = args.NC
        self.generate_kwargs = dict(
            max_new_tokens=512,
            return_dict_in_generate=True,
            output_scores=True
        )
        self.output_verbalizer = args.output_verbalizer
        self.num_docs = args.num_docs

    def forward(self, input):
        generated_ids = self.model.generate(**input, **self.generate_kwargs)
        return generated_ids

    def create_raw_result(self, ids, preds, scores):
        result = {}
        for k,v in ids.items():
            result = {k : {}}
            for i,v in enumerate(v):
                result[k][v] = {'pred':preds[i], 'score':scores[i]}
        return result

    def predict_step(self, batch, batch_idx):

        answers = [_normalize_answer(a[0]) for a in batch[2]]
        total_docs = [d[0] for d in batch[0]]
        query = batch[1][0]
        ids = {q_id : [doc_id[0] for doc_id in v] for q_id,v in batch[3].items()}
        answer_scores, preds = [], []
        for i in range(int(len(total_docs)/self.batch_size)):
            docs = total_docs[i*self.batch_size:(i+1)*self.batch_size]
            inputs = [self.template.format(p=self.prompt, d=doc, q=query) for doc in docs]
            input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device)
            # embed()
            outputs = self(input)
            # embed();exit(0)
            if self.NC:
                rel_scores = self._noisy_channel(docs, query)
                scores = self._calculate_score(outputs)
                answer_scores += [s * r for s, r in zip(scores, rel_scores)]
            else:
                answer_scores += self._calculate_score(outputs)

            preds += self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        if self.CoT or 'opt' in self.model_name:
            preds = [_normalize_answer(p.split(self.output_verbalizer)[-1]) for p in preds]
        else:
            preds = [_normalize_answer(p) for p in preds]

        raw_result = self.create_raw_result(ids, preds, answer_scores)

        if self.UC:
            preds = np.array(preds)
            answer_scores = np.array(answer_scores)
            index = preds != "unanswerable"
            preds = preds[index]
            if len(preds) == 0:
                return (0, -1, "unanswerable", raw_result)
            else:
                answer_scores = answer_scores[index]
                score = max(answer_scores)
                pred = preds[np.where(answer_scores == score)[0][0]]
        else:
            score = max(answer_scores)
            pred = preds[answer_scores.index(score)]

        acc = self._accuracy(answers, pred)
        return (acc, score, pred, raw_result)

    def _calculate_score(self, outputs):
        if "t5" in self.model_name or "T0" in self.model_name:
            scores = torch.stack(outputs.scores).transpose(0,1)
            results = []
            for i in range(outputs.sequences.shape[0]):
                result = []
                pred = self.tokenizer.batch_decode([outputs.sequences[i]],skip_special_tokens=True)
                seq_len = len(self.tokenizer(pred[0]).input_ids)
                if self.CoT:
                    ans_len = len(self.tokenizer(pred[0].split(self.output_verbalizer)[-1]).input_ids)
                else:
                    ans_len = seq_len
                for j in scores[i,seq_len-ans_len:seq_len]:
                    s = topk(log_softmax(j.unsqueeze(0)), 1).values[0][0]
                    result.append(float(s))
                results.append(exp(sum(result)/len(result)))
        elif "opt" in self.model_name:
            scores = torch.stack(outputs.scores).transpose(0,1)
            results = []
            for i in range(outputs.sequences.shape[0]):
                result = []
                for j in scores[i]:
                    s,v = topk(log_softmax(j.unsqueeze(0)), 1)
                    result.append(float(s[0][0]))
                    if int(v[0][0]) == 2:
                        break
                results.append(exp(sum(result)/len(result)))
        return results

    def _noisy_channel(self, docs, query):
        template = "Please write a question based on the following context\n\nContext: {d}\nQuestion:"
        inputs = [template.format(d=doc) for doc in docs]
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(self.device).input_ids
        labels = [query] * len(docs)
        labels = self.tokenizer(labels, padding=True, truncation=True, return_tensors="pt").to(self.device).input_ids
        if "t5" in self.model_name:
            rel_scores = []
            outputs = self.model(input_ids=inputs, labels=labels)
            logits = outputs.logits
            for i in range(len(docs)):
                log_softmax = torch.nn.functional.log_softmax(logits[i])
                nll = -log_softmax.gather(1, labels[i].unsqueeze(0).transpose(0,1))
                avg_nll = torch.sum(nll, dim=0)
                rel_scores.append(float(-avg_nll)/ float(labels[i].shape[0]))
            rel_scores = torch.nn.functional.softmax(torch.tensor(rel_scores), dim=0).tolist()
            return rel_scores
        elif "opt" in self.model_name:
            pass


    def _accuracy(self, golds, pred):
        cor = max([gold == pred for gold in golds])
        return cor

    def get_dataloader(self, data_path):
        dataset = QASDataset(data_path)
        return DataLoader(dataset, batch_size=1)

    def get_dataloader(self, corpus, queries, qrels, workers):
        dataset = Retrieved_Dataset(corpus, queries, qrels, self.num_docs)
        return DataLoader(dataset, batch_size=1, num_workers=workers)

