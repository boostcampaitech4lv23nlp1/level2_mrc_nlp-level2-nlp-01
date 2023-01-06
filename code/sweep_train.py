import logging
import os
import sys
from typing import NoReturn
import shutil
import wandb


from omegaconf import OmegaConf
from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    DataCollatorForSeq2Seq, # 다른 시퀀스 length을 가진 input들을 합쳐줘서 gpu에서 pair computing이 쉽게 만들어 준다.
    Seq2SeqTrainer, # 
    AutoTokenizer,
    Seq2SeqTrainingArguments, #
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    T5TokenizerFast, 
    T5ForConditionalGeneration
)
import numpy as np

from utils_qa import check_no_error, postprocess_qa_predictions
from preprocess import prepare_train_features, prepare_validation_features

logger = logging.getLogger(__name__)


def main(cfg, sweep_cfg):
    output_dir = cfg.train.path.output_dir
    dataset_name = cfg.train.path.dataset_name
    delete_exist_output = cfg.train.path.delete_exist_output

    do_train = cfg.train.stage.do_train
    do_eval = cfg.train.stage.do_eval
    overwrite_cache = cfg.train.stage.overwrite_cache

    model_name_or_path = cfg.train.model.model_name_or_path
    tokenizer_name = cfg.train.model.tokenizer_name
    label_smoothing_factor = cfg.train.model.label_smoothing_factor
    num_train_epochs = cfg.train.model.num_train_epochs
    per_device_train_batch_size = cfg.train.model.per_device_train_batch_size
    learning_rate = cfg.train.model.learning_rate
    warmup_steps = cfg.train.model.warmup_steps
    weight_decay = cfg.train.model.weight_decay
    fp16 = cfg.train.model.fp16

    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    parser = HfArgumentParser(
        (
            ModelArguments, 
            DataTrainingArguments, 
            TrainingArguments
        )
    )
    if delete_exist_output is True:
        if os.path.exists(output_dir):
            flag = input(f"정말 {output_dir} 내의 모든 파일을 삭제하시겠습니까? (yes) >> ")
            if flag == 'yes':
                shutil.rmtree(output_dir)
            else:
                print('기존 output을 삭제하지 않습니다.')

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(['--output_dir', output_dir])
    print(model_args.model_name_or_path)

    # ------------------------ hyper parameter 설정 ------------------------ #
    data_args.dataset_name = dataset_name
    data_args.overwrite_cache = overwrite_cache

    model_args.model_name_or_path = model_name_or_path
    model_args.tokenizer_name = tokenizer_name

    training_args.do_train = do_train
    training_args.do_eval = do_eval
    training_args.label_smoothing_factor = label_smoothing_factor
    training_args.num_train_epochs = num_train_epochs
    training_args.per_device_train_batch_size = per_device_train_batch_size
    training_args.learning_rate = learning_rate
    training_args.warmup_steps = warmup_steps
    training_args.weight_decay = weight_decay
    training_args.fp16 = fp16

    training_args.logging_strategy = 'epoch'
    training_args.evaluation_strategy = 'epoch'
    

    # ----------------------------------------------------------------------- #
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")
    

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # dataset를 불러옵니다.
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    if cfg.reader.mode.generation is False:
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
            if model_args.tokenizer_name is not None
            else model_args.model_name_or_path,
            # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
            # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
            # rust version이 비교적 속도가 빠릅니다.
            use_fast=True,
            stride=data_args.doc_stride,
        )
        model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if cfg.reader.mode.generation and (training_args.do_train or training_args.do_eval):
        print('Generation based MRC training')
        run_mrc_based_generation(data_args, training_args, model_args, datasets, tokenizer, model)
    elif training_args.do_train or training_args.do_eval:
        print('Extraction based MRC training')
        run_mrc_based_extraction(data_args, training_args, model_args, datasets, tokenizer, model)


def run_mrc_based_extraction(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    sweep_cfg
) -> NoReturn:
    try:
        wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    sweep_cfg = OmegaConf.to_object(sweep_cfg)
    '''
    sweep_cfg = {
        'method': 'random',
        'parameters':{
            'learning_rate':{
                'distribution': 'uniform',
                'min': 1e-5,
                'max': 1e-4
            },
            'label_smoothing_factor':{
                'values': [0., 0.1],
            },
            'warmup_steps': {
                'values': [250, 500],
            },
            'weight_decay': {
                'values': [0.01, 0.005]
            }
        },
        'metric':{
            'name': 'train/loss',
            'goal': 'minimize'
        }
    }
    '''
    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project='sweep'
    )

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
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

    # Train preprocessing / 전처리를 진행합니다.
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            function=lambda x: prepare_train_features(x, tokenizer=tokenizer, pad_on_right=pad_on_right,
                                                    context_column_name=context_column_name, question_column_name=question_column_name,
                                                    answer_column_name=answer_column_name,
                                                    data_args=data_args, max_seq_length=max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
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

    # define sweep train
    def sweep_train(config=None):
        wandb.init(project='sweep', name='tuning', config=config)
        config = wandb.config
        
        print('label_smoothing_factor :', config.label_smoothing_factor)
        print('learning_rate :', config.learning_rate)
        print('warmup_steps :', config.warmup_steps)
        print('weight_decay :', config.weight_decay)

        training_args.label_smoothing_factor = config.label_smoothing_factor
        training_args.learning_rate = config.learning_rate
        training_args.warmup_steps = config.warmup_steps
        training_args.weight_decay = config.weight_decay


        # Data collator
        # flag가 True이면 이미 max length로 padding된 상태입니다.
        # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )

        # Post-processing:
        def post_processing_function(examples, features, predictions, training_args):
            # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
            predictions = postprocess_qa_predictions(
                examples=examples,          # 전처리 이전 데이터셋
                features=features,          # 전처리 후 데이터셋
                predictions=predictions,    # 모델의 예측값: start logits과 end logits를 나타냄
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

        def compute_metrics(p: EvalPrediction):
            return metric.compute(predictions=p.predictions, references=p.label_ids)

        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            eval_examples=datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=post_processing_function,
            compute_metrics=compute_metrics,
        )

        # Training
        if training_args.do_train:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
            else:
                checkpoint = None
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # State 저장
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()

            metrics["eval_samples"] = len(eval_dataset)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=10
    )

def run_mrc_based_generation(
    data_args: DataTrainingArguments,
    training_args: Seq2SeqTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
    ) -> NoReturn:

    # sweep을 위한 설정
    try:
        wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    sweep_cfg = OmegaConf.to_object(sweep_cfg)

    '''
    sweep_cfg = {
        'method': 'random',
        'parameters':{
            'learning_rate':{
                'distribution': 'uniform',
                'min': 1e-5,
                'max': 1e-4
            },
            'label_smoothing_factor':{
                'values': [0., 0.1],
            },
            'warmup_steps': {
                'values': [250, 500],
            },
            'weight_decay': {
                'values': [0.01, 0.005]
            }
        },
        'metric':{
            'name': 'train/loss',
            'goal': 'minimize'
        }
    }
    '''

    sweep_id = wandb.sweep(
        sweep=sweep_cfg,
        project='sweep'
    )

    # dataset을 가져옵니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    # 오류가 있는지 확인합니다.
    last_checkpoint, _ = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # 모델 입력에 사용할 전처리 함수
    def preprocess_function(examples):
        inputs = [f"question: {q} </s> context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
        targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
        model_inputs = tokenizer(
            inputs,
            max_length=cfg.reader.generation.max_source_length,
            padding=cfg.reader.generation.padding,
            truncation=True
        )
        # targets(label)을 위해 tokenizer 설정
        labels = tokenizer(
            text_target=targets,
            max_length=cfg.reader.generation.max_target_length,
            padding=cfg.reader.generation.padding,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"] 
        model_inputs["example_id"] = []
        for i in range(len(model_inputs["labels"])):
            model_inputs["example_id"].append(examples["id"][i])
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        )
    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # sweep train함수 정의
    def sweep_train(config=None):
        wandb.init(project='sweep', name='tuning', config=config)
        config = wandb.config
        
        print('label_smoothing_factor :', config.label_smoothing_factor)
        print('learning_rate :', config.learning_rate)
        print('warmup_steps :', config.warmup_steps)
        print('weight_decay :', config.weight_decay)

        training_args.label_smoothing_factor = config.label_smoothing_factor
        training_args.learning_rate = config.learning_rate
        training_args.warmup_steps = config.warmup_steps
        training_args.weight_decay = config.weight_decay


        # Data collator
        label_pad_token_id = tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
        )

        metric = load_metric("rouge")

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
            return result.mid

        training_args.predict_with_generate=True
        training_args.save_total_limit=2

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        # Training
        if training_args.do_train:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
            else:
                checkpoint = None
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # State 저장
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )
        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(
                max_length=cfg.reader.generation.max_target_length,
                num_beams=cfg.reader.generation.num_beams,
                metric_key_prefix="eval"
            )
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)

    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=10
    )

    # validation 정답 확인을 위한 함수
    def generarate_answer(sample):
        inputs = f'question: {sample["question"]} </s> context: {sample["context"]} </s>'
        print(inputs.split('context:')[0])
        print("context: "+ inputs.split('context:')[1])
        sample = tokenizer(inputs, max_length=cfg.reader.generation.max_source_length, padding=cfg.reader.generation.padding, truncation=True, return_tensors='pt')
        sample = sample.to("cuda:0")
        # sample = sample.to('cpu')
        outputs = model.generate(**sample, max_length=cfg.reader.generation.max_target_length, num_beams=cfg.reader.generation.num_beams)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        pred = "\n".join(nltk.sent_tokenize(pred))
        answer = 'answer :' + pred
        return answer

    
    np.random.seed(seed=7777) 

    for i in np.random.randint(0, len(datasets["validation"]), 5):
        print(generarate_answer(datasets["validation"][int(i)]))
        print("=" * 8)



if __name__ == "__main__":
    # configuation
    config_name = 'roberta-large_config'
    cfg = OmegaConf.load(f'./conf/reader/{config_name}.yaml')

    wandb_config_name = 'wandb_config'
    sweep_cfg = OmegaConf.load(f'./conf/wandb_sweep/{wandb_config_name}.yaml')

    main(cfg, sweep_cfg)
