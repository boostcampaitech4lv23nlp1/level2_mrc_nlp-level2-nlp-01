from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import json
import time
import os
from contextlib import contextmanager
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import TrainingArguments
from datasets import load_from_disk
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BertEncoder(BertPreTrainedModel):

    def __init__(self,config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
      
    def forward(self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output
    

class DenseRetrieval():
    def __init__(
        self, 
        args,
        tokenizer,
        p_encoder,
        q_encoder,
        dataset_path: Optional[str] = "../data/wikipedia_documents.json", 
    ):
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.args = args
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        dataset = load_from_disk("/opt/ml/input/data/train_dataset")
        self.wiki_corpus = self.load_data("../data/wikipedia_documents.json")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        p_with_neg = self.prepare_in_batch_negative(train_dataset)
        q_seqs = tokenizer(
            train_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, 10, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, 10, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, 10, max_len)
        self.train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )
        p_with_neg = self.prepare_in_batch_negative(eval_dataset)
        q_seqs = tokenizer(
            eval_dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, 10, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, 10, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, 10, max_len)
        self.eval_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

    def load_data(self,dataset_path):
        with open(dataset_path, "r") as f:
            wiki = json.load(f)

        wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        wiki_corpus = [wiki_texts[i] for i in range(len(wiki_texts))]
        return wiki_corpus

    def prepare_in_batch_negative(self,train_dataset):
        num_neg = 9
        corpus = np.array(train_dataset["context"])
        p_with_neg = []

        for c in train_dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)
                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]
                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
        return p_with_neg
    def train(self):
        args = self.args
        train_dataset = self.train_dataset
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        num_neg = 9
        batch_size = args.per_device_train_batch_size
    
        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = p_encoder(**p_inputs)
                    # (batch_size, emb_dim)  
                    q_outputs = q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, num_neg + 1, -1)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    q_encoder.zero_grad()
                    p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

        return p_encoder, q_encoder

 
    
    def get_relevant_doc(self, queries, k=2, args=None, p_encoder=None, q_encoder=None, tokenizer=None, wiki_corpus =None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder
        
        if wiki_corpus is None :
            # wiki_corpus = self.wiki_corpus
            dataset_path = "../data/wikipedia_documents.json"
            with open(dataset_path, "r") as f:
                wiki = json.load(f)

            wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            wiki_corpus = [wiki_texts[i] for i in range(len(wiki_texts))]
        
        if tokenizer is None : 
            tokenizer = self.tokenizer
        
        valid_seqs = tokenizer(wiki_corpus, padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=8)

        
        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()
            results = []
            with tqdm(queries, unit="query") as tepoch:
                for query in tepoch:
                    q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
                    q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

                    p_embs = []
            
                    for batch in self.passage_dataloader:

                        batch = tuple(t.to(args.device) for t in batch)
                        p_inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2]
                        }
                        p_emb = p_encoder(**p_inputs).to('cpu')
                        p_embs.append(p_emb)

                    p_embs = torch.stack(p_embs, dim=0).view(len(wiki_corpus), -1)  # (num_passage, emb_dim)

                    dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
                    rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
                    idx = rank[:k].tolist()
                    result = wiki_corpus[idx]
                    results.append(result)
        return results

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]['_source']['document_text'])

            return ([doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환
            total = []

            with timer("query exhaustive search"):  
                docs = self.get_relevant_doc(query_or_dataset["question"], k=topk)
                print(docs)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")):
                # retrieved_context 구하는 부분 수정
                retrieved_context = []
                for i in range(topk):
                    retrieved_context.append(docs[idx][i])

                tmp = {
                    # Query와 해당 id를 반환
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(retrieved_context),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)
            
            cqas = pd.DataFrame(total)
            return cqas

def main():
    full_ds = load_from_disk("../data/train_dataset")['train']
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    print(len(full_ds))
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()
    if os.path.exists('./dense_p_encoder') and os.path.exists('./dense_q_encoder'):
        p_encoder = torch.load('./dense_p_encoder')
        q_encoder = torch.load('./dense_q_encoder')
        retriever = DenseRetrieval(args= args, tokenizer= tokenizer, p_encoder= p_encoder, q_encoder= q_encoder)

    else:
        retriever = DenseRetrieval(args= args, tokenizer= tokenizer, p_encoder= p_encoder, q_encoder= q_encoder)
        p_encoder, q_encoder = retriever.train()
        torch.save(p_encoder, "./dense_p_encoder")
        torch.save(q_encoder, "./dense_q_encoder")
    
    
    df = retriever.retrieve(full_ds, topk=10)

    print(df)
    df["correct"] = [original_context in context for original_context,context in zip(df["original_context"],df["context"])]
    print(
        "correct retrieval result by exhaustive search",
        f"{df['correct'].sum()}/{len(df)}",
        df["correct"].sum() / len(df),
    )
if __name__ == "__main__":
    main()