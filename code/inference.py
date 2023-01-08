"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.
대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys
from typing import Callable, Dict, List, NoReturn, Tuple
import json
import collections

import pickle
import numpy as np
import nltk
from tqdm import tqdm
from elastic_retrieval import ElasticRetrieval
from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_from_disk,
    load_metric,
)
from retrieval import SparseRetrieval, BM25
from omegaconf import OmegaConf
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorForSeq2Seq, # 다른 시퀀스 length을 가진 input들을 합쳐줘서 gpu에서 pair computing이 쉽게 만들어 준다.
    Seq2SeqTrainer, # 
    Seq2SeqTrainingArguments, #
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    T5TokenizerFast, 
    T5ForConditionalGeneration
)
from utils_qa import check_no_error, postprocess_qa_predictions
from preprocess import prepare_validation_features

logger = logging.getLogger(__name__)


def main(cfg):
    output_dir = cfg.test.path.output_dir
    dataset_name = cfg.test.path.dataset_name

    do_train = cfg.test.stage.do_train
    do_eval = cfg.test.stage.do_eval
    do_predict = cfg.test.stage.do_predict

    model_name_or_path = cfg.test.model.model_name_or_path

    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(['--output_dir', output_dir])

    # ------------------------ hyper parameter 설정 ------------------------ #
    data_args.dataset_name = dataset_name
    model_args.model_name_or_path = model_name_or_path

    training_args.do_train = do_train
    training_args.do_eval = do_eval
    training_args.do_predict = do_predict

    
    # ----------------------------------------------------------------------- #


    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    

    if cfg.reader.mode.generation==False:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            use_fast=True,
                )
        model = AutoModelForQuestionAnswering.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        tokenizer = T5TokenizerFast.from_pretrained(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path,
            use_fast=True,
                )
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )


    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )

    # eval or predict mrc model
    if cfg.reader.mode.generation and (training_args.do_eval or training_args.do_predict):
        print('Generation based MRC evaluation')
        run_mrc_based_generation(data_args, training_args, model_args, datasets, tokenizer, model)
    elif training_args.do_eval or training_args.do_predict:
        print('Extraction based MRC evaluation')
        run_mrc_based_extraction(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "/opt/ml/input/data/",
    context_path: str = "wikipedia_documents.json",
    ) -> DatasetDict:

    if data_args.use_elastic: 
        retriever = ElasticRetrieval(data_args.elastic_index_name)
    # Query에 맞는 Passage들을 Retrieval 합니다.
    else:
        retriever = BM25(tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path)
        retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc_based_extraction(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    ) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        function=lambda x: prepare_validation_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                    context_column_name=context_column_name, question_column_name=question_column_name,
                                                    answer_column_name=answer_column_name,
                                                    data_args=data_args, max_seq_length=max_seq_length),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions
        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]

            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

def run_mrc_based_generation(
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    ) -> NoReturn:

    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    eval_dataset = datasets["validation"]

    # print(eval_dataset)

    # preprocessing
    def preprocess_function(examples):

        print(len(examples['id']))
        inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]

        model_inputs = tokenizer(
            inputs,
            max_length=cfg.reader.generation.max_source_length,
            padding=cfg.reader.generation.padding,
            truncation=True
        )

        model_inputs["example_id"] = []
        for i in range(len(examples['id'])): # 600 (dataset length)
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            )

    label_pad_token_id = tokenizer.pad_token_id

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )

    metric = load_metric("squad")

    def postprocess_text(preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
            
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result

    training_args.generation_max_length = cfg.reader.generation.max_target_length
    training_args.generation_num_beams = cfg.reader.generation.num_beams
    training_args.predict_with_generate=True

    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    def generarate_answer(org_dataset, eval_dataset):
        all_predictions = collections.OrderedDict()
        for i in tqdm(range(len(org_dataset))):
            sample = org_dataset[i]    
            inputs = f'question: {sample["question"]} </s> context: {sample["context"]} </s>'
            sample = tokenizer(inputs, max_length=cfg.reader.generation.max_source_length, padding=cfg.reader.generation.padding, truncation=True, return_tensors='pt')
            sample = sample.to("cuda:0")
            outputs = model.generate(**sample, max_length=cfg.reader.generation.max_target_length, num_beams=cfg.reader.generation.num_beams)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            pred = "\n".join(nltk.sent_tokenize(pred))

            all_predictions[org_dataset[i]['id']] = pred

        assert len(all_predictions) == len(org_dataset)
        return all_predictions


    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        prediction_file = training_args.output_dir + '/prediction.json'

        all_predictions = generarate_answer(datasets["validation"], eval_dataset)
        

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    # configuation
    config_name = 'roberta-large_config'
    cfg = OmegaConf.load(f'./conf/reader/{config_name}.yaml')

    main(cfg)
