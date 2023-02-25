from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import json
from torch import nn

class QASDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        self.context, self.question, self.answer = [],[], []
        with open(file_path, 'r') as f:
            data = json.load(f)
        for d in data:
            self.question.append(d['question'])
            self.context.append(d['positive_ctxs'][0]['text'])
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
        self.model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="/home/tjrals/models")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.template = "{d}\nQuestion: {q}\nAnswer: "
        self.batch = args.batch

    def forward(self, input):
        generated_ids = self.model.generate(**input, max_length=512)
        return generated_ids

    def predict_step(self, batch, batch_idx):
        texts = [self.template.format(d=d, q=q) for d,q in zip(batch[0], batch[1])]
        input = self.tokenizer(texts, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self(input)
        preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(preds)
        exit(0)

    def get_dataloader(self, data_path):
        dataset = QASDataset(data_path)
        return DataLoader(dataset, batch_size=int(self.batch))

