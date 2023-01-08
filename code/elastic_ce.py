from email.policy import default
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm
from elastic_setting import *
from cross_encoder import CrossEncoder

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

    
class ElasticRetrieval:
    def __init__(self, INDEX_NAME):
        self.es, self.index_name = es_setting(index_name=INDEX_NAME) 
        
        
    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]['_source']['document_text'])

            return (doc_scores, [doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
        
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")):
                # retrieved_context 구하는 부분 수정
                retrieved_context = []
                for i in range(min(topk, len(docs[idx]))):
                    retrieved_context.append(docs[idx][i]['_source']['document_text'])
                    
                
                tmp = {
                    # Query와 해당 id를 반환
                    "question": example["question"],
                    "id": example["id"],
                    "context": retrieved_context,
                }

                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                
                total.append(tmp)

            results = self.rerank(total)
            cqas = pd.DataFrame(results)
            return cqas
    
    
    def rerank(self, results) :
        totals = []
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        for result in results:
            sentence_pairs = []

            for context in result['context']:
                sentence_pairs.append([result['question'], context])
            rerank_scores = np.array([float(score) for score in cross_encoder.predict(sentence_pairs, batch_size=128)])
            sort_info = rerank_scores.argsort()

            new_context = np.array(result['context'])[sort_info][::-1].tolist()
            result['context'] = new_context[:40]
            totals.append(result)
        return totals
        

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        doc_score = []
        doc_index = []
        res = es_search(self.es, self.index_name, query, k)
        docs = res['hits']['hits']

        for hit in docs:
            doc_score.append(hit['_score'])
            doc_index.append(hit['_id'])
            print("Doc ID: %3r  Score: %5.2f" % (hit['_id'], hit['_score']))

        return doc_score, doc_index, docs
    
    
    def get_relevant_doc_bulk(self, queries: List, k: Optional[int] = 1) -> Tuple[List, List]:
        total_docs = []
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score = []
            doc_index = []
            res = es_search(self.es, self.index_name, query, k)
            docs = res['hits']['hits']

            for hit in docs:
                doc_score.append(hit['_score'])
                doc_indices.append(hit['_id'])

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
            total_docs.append(docs)

        return doc_scores, doc_indices, total_docs
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str, help="")
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--index_name", default="origin-wiki", type=str, help="Define the test name")

    args = parser.parse_args()

    # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)
    print(len(org_dataset["train"]),len(org_dataset["validation"]))

    retriever = ElasticRetrieval(args.index_name)
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.use_faiss:

        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve_faiss(query)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=100)
            print(df)
            df["correct"] = [original_context in context for original_context,context in zip(df["original_context"],df["context"])]
            print(
                "correct retrieval result by exhaustive search",
                f"{df['correct'].sum()}/{len(df)}",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query,1)