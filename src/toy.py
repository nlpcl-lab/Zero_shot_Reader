from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
from src.util import _normalize_answer
import pytorch_lightning as pl
import json
from spacy.lang.en import English
from torch.nn.functional import log_softmax
from torch import topk
from tqdm import tqdm
from math import exp
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
            # elif len(d['positive_ctxs']) >=3:
            #     self.context.append(d['positive_ctxs'][2]['text'])
            # elif len(d['positive_ctxs']) >=2:
            #     self.context.append(d['positive_ctxs'][1]['text'])
            # else:
            # self.context.append(d['negative_ctxs'][0]['text'])
            # self.context.append("")
            # self.context.append((d['positive_ctxs'][0]['text'], d['positive_ctxs'][1]['text']))
            self.answer.append(d['answers'])

    def __len__(self):
        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answer)
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.question[i], self.answer[i]

class Retrieved_Dataset(Dataset):
    def __init__(self, corpus, queries, qrels):
        super().__init__()
        self.context, self.question, self.answer = [], [], []
        for q,documents in qrels.items():
            self.question.append(queries[q]['text'])
            self.answer.append(queries[q]['answer'])
            self.context.append([corpus[d]['text'] for d in documents.keys()][:9])

    def __len__(self):
        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answer)
        return len(self.context)

    def __getitem__(self, i):
        return self.context[i], self.question[i], self.answer[i]
    #

class Phrase_QASDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        nlp = English()
        nlp.add_pipe("sentencizer")

        self.context, self.question, self.answer = [],[], []
        with open(file_path, 'r') as f:
            data = json.load(f)
        for d in tqdm(data):
            self.question.append(d['question'])
            doc = nlp(d['positive_ctxs'][0]['text'])
            sents_list = [sent.text for sent in doc.sents]
            self.context.append(sents_list)
            self.answer.append(d['answers'])

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
        if "T0" not in args.model:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="./models/")
            self.tokenizer = AutoTokenizer.from_pretrained(args.model)
            self.template = "Passage: {d}\nQuestion: {q}\nAnswer: "
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("./models/T0_3B")
            self.tokenizer = T5Tokenizer.from_pretrained("./models/T0_3B")
            self.template = "Please answer the question based on the passage.\n\nPassage: {d}\n\nQuestion: {q}"
        self.batch_size = args.batch
        self.cs = args.cs

    def forward(self, input):
        if 'T0' not in self.model_name:
            max_length = input.input_ids.shape[1]+10
        else:
            max_length = 10
        generated_ids = self.model.generate(input.input_ids, max_length=max_length,return_dict_in_generate=True, output_scores=True)
        return generated_ids


    def predict_step(self, batch, batch_idx):
        texts = [self.template.format(d="", q=q) for q in batch[1]]
        input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self(input)
        no_score = self._calculate_score(generated_ids)
        answers = [_normalize_answer(a) for a in batch[2][0]]

        scores, preds = [], []
        for d in batch[0]:
            texts = [self.template.format(d=d, q=q) for q in (batch[1])]
            input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
            generated_ids = self(input)
            score = self._calculate_score(generated_ids)
            pred = self.tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
            pred = [_normalize_answer(p.split("Answer: ")[-1]) for p in pred]
            scores.append(score)
            preds.append(pred)
            if self.cs:
                if score - no_score > 0.4:
                    acc = self._accuracy(answers, pred)
                    result = (acc, score - no_score)
                    return result

        score = max(scores)
        pred = preds[scores.index(score)]
        acc = self._accuracy(answers, pred)
        result = (acc, score - no_score)
        return result


    def _calculate_score(self, outputs):
        scores = []
        for i in range(1,outputs.sequences.shape[1]):
            s = topk(log_softmax(outputs.scores[i-1]),1).values[0][0]
            scores.append(float(s))
        return exp(sum(scores)/len(scores))

    def _accuracy(self, golds, preds):
        total = len(golds)
        cor = sum([int(g in p) for g, p in zip(golds, preds)])
        return cor/total

    def get_dataloader(self, data_path):
        dataset = QASDataset(data_path)
        return DataLoader(dataset, batch_size=int(self.batch_size))

    def get_dataloader(self, corpus, queries, qrels):
        dataset = Retrieved_Dataset(corpus, queries, qrels)
        return DataLoader(dataset, batch_size=int(self.batch_size))


class Phrase_Reader(Reader):
    def __init__(self, args):
        super().__init__(args)

    def test_step(self, batch, batch_idx):
        documents = batch[0][0]
        question = batch[1][0]
        texts = [self.template.format(d=d, q=question) for d in documents]
        preds = {}
        for t in texts:
            input = self.tokenizer(t, padding=True, return_tensors="pt").to(self.device)
            outputs = self(input)
            score = self._calculate_score(outputs)
            pred = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            if pred not in preds:
                preds[pred] = score
            elif preds[pred] < score:
                preds[pred] = score
        sorted_preds = sorted(preds.items(), key=lambda item: item[1], reverse=True)
        preds = [_normalize_answer(sorted_preds[0][0])]

        # preds = self.tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)
        # preds = [_normalize_answer(p.split("Answer: ")[-1]) for p in preds]
        answers = [_normalize_answer(a) for a in batch[2][0]]
        acc = self._accuracy(answers, preds)
        self.log("Test_Acc", acc,batch_size=self.batch_size)

    def get_dataloader(self, data_path):
        dataset = Phrase_QASDataset(data_path)
        return DataLoader(dataset, batch_size=int(self.batch_size))

