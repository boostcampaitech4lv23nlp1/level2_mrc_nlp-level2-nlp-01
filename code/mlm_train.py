import logging
import os
import sys
from typing import NoReturn
import shutil
import utils
import wandb

from omegaconf import OmegaConf
from arguments import DataTrainingArguments, ModelArguments
from datasets import Dataset, DatasetDict, load_from_disk, load_metric, load_dataset
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
)
from utils_qa import check_no_error, postprocess_qa_predictions
from preprocess import prepare_train_features, prepare_validation_features

import nltk
nltk.download('punkt')

logger = logging.getLogger(__name__)


def main(cfg):
    '''
        TODO 1: 데이터셋을 불러옵니다.
            1) 기존 데이터셋
            2) korquad 데이터셋
    '''
    datasets = load_dataset('squad_kor_v1')
    print(datasets)

    column_names = datasets['train'].column_names
    context_column_name = "context" if "context" in column_names else column_names[1]

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='klue/roberta-large',
        use_fast=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )

    def preprocessing(x):
        tokenized_data = tokenizer(
            x[context_column_name],
            max_length=384,
            stride=128,
            padding=False
        )
        return tokenized_data

    train_datasets = datasets['train']
    train_datasets = train_datasets.map(
        function=preprocessing,
        batched=True,
        remove_columns=column_names
    )
    
    validation_datasets = datasets['validation']
    validation_datasets = validation_datasets.map(
        function=preprocessing,
        batched=True,
        remove_columns=column_names
    )
    model = AutoModelForMaskedLM.from_pretrained('klue/roberta-large')

    # ------------------------------------------------------------------------------------ # 

    try:
        wandb.login(key='4c0a01eaa2bd589d64c5297c5bc806182d126350')
    except:
        anony = "must"
        print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')

    wandb.init(project="project", name= "test")

    training_args = TrainingArguments(
        output_dir=cfg.train.path.output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=validation_datasets,
        data_collator=data_collator,
    )

    print('pretraining start')
    trainer.train()

    
if __name__ == "__main__":
    # configuation
    config_name = 'roberta-large_config'
    cfg = OmegaConf.load(f'./conf/reader/{config_name}.yaml')

    main(cfg)
