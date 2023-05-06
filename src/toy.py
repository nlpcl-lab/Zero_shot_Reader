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
import torch
from IPython import embed

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
        self.context, self.question, self.answer = [], [], []
        for q,documents in qrels.items():
            self.question.append(queries[q]['text'])
            self.answer.append(queries[q]['answer'])
            self.context.append([corpus[d]['text'] for d in documents.keys()][:num_docs])

    def __len__(self):
        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answer)
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.question[i], self.answer[i]

class Reader(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model
        self.uncertain = args.uncertain
        if "opt" in args.model:
            self.model = AutoModelForCausalLM.from_pretrained("../models/"+args.model)
            self.tokenizer = AutoTokenizer.from_pretrained("../models/"+args.model)
            if args.uncertain == 1 and args.uncertain == 2:
                self.template = "Read the following context and answer the question. If you can't find an answer, return Unanswerable\n\nContext: {d}\nQuestion: {q}\nAnswer:"
            else:
                self.template = "Read the following context and answer the question.\n\nContext: {d}\nQuestion: {q}\nAnswer:"
        elif "t5" in args.model:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("../models/"+args.model)
            self.tokenizer = AutoTokenizer.from_pretrained("../models/"+args.model)
            if args.uncertain == 1 and args.uncertain == 2:
                self.template = 'Read the following context and answer the question. If you can\'t find an answer, return "unanswerable"\n\nContext: {d}\nQuestion: {q}\nAnswer:'
            elif args.uncertain == 4:
                self.template = "Read the following context and answer the question with reasoning step-by-step.\n\nContext: {d}\nQuestion: {q}\nAnswer:"
            else:
                self.template = "Read the following context and answer the question.\n\nContext: {d}\nQuestion: {q}\nAnswer:"
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("../models/"+ args.model)
            self.tokenizer = T5Tokenizer.from_pretrained("../models/" + args.model)
            if args.uncertain == 1 and args.uncertain == 2:
                self.template = 'Read the following contest and answer the question. If you can\'t find an answer, return "unanswerable"\n\nContext: {d}\nQuestion: {q}\nAnswer:'
            else:
                self.template = "Read the following context and answer the question with reasoning step-by-step.\n\nContext: {d}\nQuestion: {q}\nAnswer:"
        self.batch_size = args.batch
        self.cs = args.cs
        self.num_docs = args.num_docs
        self.threshold = args.threshold

    def forward(self, input):
        if "opt" in self.model_name:
            generated_ids = self.model.generate(input.input_ids, max_length=input.input_ids.shape[1]+10, return_dict_in_generate=True, output_scores=True)
        elif "t5" in self.model_name:
            generated_ids = self.model.generate(input.input_ids, max_length= 512, return_dict_in_generate=True, output_scores=True)
        return generated_ids

    def get_score(self, doc, query):
        template = "Please write a question based on the following context\n\nContext: {d}\nQuestion:"
        if "opt" in self.model_name:
            prompt_len = self.tokenizer(template.format(d=doc), return_tensors="pt").input_ids.shape[1]
            input = self.tokenizer(template.format(d=doc) + " " + query, return_tensors="pt").to(self.device)
            query_token = input.input_ids[:,prompt_len:]
            logits = self.model(**input).logits
            log_softmax = torch.nn.functional.log_softmax(logits[:, prompt_len - 1:-1, :], dim=-1)
            nll = -log_softmax.gather(2, query_token.unsqueeze(2)).squeeze(2)
            avg_nll = torch.sum(nll, dim=1)
        elif "t5" in self.model_name:
            input_encoding = self.tokenizer(template.format(d=doc), return_tensors="pt").to(self.device)
            context_tensor, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            output_encoding = self.tokenizer(query, return_tensors="pt").to(self.device)
            decoder_prefix_tensor = output_encoding.input_ids
            logits = self.model(input_ids=context_tensor, attention_mask=attention_mask,
                                labels=decoder_prefix_tensor).logits
            log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = -log_softmax.gather(2, decoder_prefix_tensor.unsqueeze(2)).squeeze(2)
            avg_nll = torch.sum(nll, dim=1) / decoder_prefix_tensor.shape[1]

        return float(-avg_nll)

    def predict_step(self, batch, batch_idx):
        # texts = [self.template.format(d="", q=q) for q in batch[1]]
        # input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        # generated_ids = self(input)
        # no_score = self._calculate_score(generated_ids)
        answers = [_normalize_answer(a) for a in batch[2][0]]
        scores, preds, rel_scores = [], [], []
        for d in batch[0]:
            q = batch[1][0]
            texts = [self.template.format(d=d[0], q=q)]
            input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            generated_ids = self(input)
            if self.uncertain >= 2 and self.uncertain < 4:
                rel_score = self.get_score(d[0],q)
            pred = self.tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
            seq_len = len(self.tokenizer(pred[0].split("the final answer is ")[-1])['input_ids'])
            score = self._calculate_score(generated_ids, seq_len)
            if self.uncertain == 4:
                pred = [_normalize_answer(p.split("the final answer is")[-1]) for p in pred]
            else:
                pred = [_normalize_answer(p.split("Answer:")[-1]) for p in pred]
            if pred[0] != "unanswerable":
                scores.append(score)
                preds.append(pred)
                if self.uncertain >= 2 and self.uncertain < 4:
                    rel_scores.append(float(rel_score))
                # if self.cs:
                #     if score - no_score > self.threshold:
                #         acc = self._accuracy(answers, pred)
                #         result = (acc, score - no_score)
                #         return result
        if len(scores) == 0:
            pred = 'unanswerable'
            score = -1
        else:
            if self.uncertain >= 2 and self.uncertain < 4:
                rel_scores = torch.nn.functional.softmax(torch.tensor(rel_scores),dim=0).tolist()
                scores = [s * r for s,r in zip(scores,rel_scores)]
            score = max(scores)
            pred = preds[scores.index(score)]
        acc = self._accuracy(answers, pred)
        result = (acc, score, pred)
        return result


    def _calculate_score(self, outputs, seq_len):
        if 't5' in self.model_name:
            scores = []
            for i in outputs.scores[-1 * seq_len:]:
                s = topk(log_softmax(i),1).values[0][0]
                scores.append(float(s))
        elif 'T0' not in self.model_name:
            scores = []
            for i in outputs.scores:
                s = topk(log_softmax(i),1).values[0][0]
                scores.append(float(s))
        else:
            scores = []
            for i in range(1,outputs.sequences.shape[1]):
                s = topk(log_softmax(outputs.scores[i-1]),1).values[0][0]
                scores.append(float(s))
        return exp(sum(scores)/len(scores))

    def _accuracy(self, golds, pred):
        cor = max([gold == pred[0] for gold in golds])
        return cor

    def get_dataloader(self, data_path):
        dataset = QASDataset(data_path)
        return DataLoader(dataset, batch_size=int(self.batch_size))

    def get_dataloader(self, corpus, queries, qrels):
        dataset = Retrieved_Dataset(corpus, queries, qrels, self.num_docs)
        return DataLoader(dataset, batch_size=int(self.batch_size))

