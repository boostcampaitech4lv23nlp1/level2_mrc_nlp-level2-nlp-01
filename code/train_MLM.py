import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    HfArgumentParser,
    set_seed,
)
from datasets import load_from_disk, load_dataset
from omegaconf import OmegaConf

from arguments import ModelArguments

class MLM_Dataset:
    def __init__(
        self,
        queries,
        tokenizer,
        model_name = 'klue/roberta-large',
        max_length = 32
    ):
        self.queries = queries
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length

    def __getitem__(self, item):
        if isinstance(self.queries[item], str):
            self.queries[item] = self.tokenizer(self.queries[item],
                                                add_special_tokens = True,
                                                truncation = True,
                                                padding = 'max_length',
                                                max_length = self.max_length,
                                                return_token_type_ids = False if 'klue/roberta' in self.model_name else None,
                                                return_special_tokens_mask = True
                                                )
        return self.queries[item]

    def __len__(self):
        return len(self.queries)


def main(cfg):
    output_dir = cfg.train.dir.output_dir
    dataset_dir = cfg.train.dir.data_dir

    do_eval = cfg.train.stage.do_eval
    save_steps = cfg.train.model.save_steps
    eval_steps = cfg.train.model.eval_steps
    logging_steps = cfg.train.model.logging_steps
    save_total_limit = cfg.train.model.save_total_limit

    model_name = cfg.train.model.model_name
    num_train_epochs = cfg.train.model.num_train_epochs
    learning_rate = cfg.train.model.learning_rate
    per_device_train_batch_size = cfg.train.model.per_device_train_batch_size
    per_device_eval_batch_size = cfg.train.model.per_device_eval_batch_size
    weight_decay = cfg.train.model.weight_decay
    warmup_steps = cfg.train.model.warmup_steps
    max_seq_length = cfg.train.model.max_seq_length
    masking_probability = cfg.train.model.masking_probability
    do_whole_word_mask = cfg.train.model.do_whole_word_mask

    parser = HfArgumentParser(
        (
            ModelArguments,
            TrainingArguments
        )
    )
    model_args, training_args = parser.parse_args_into_dataclasses(['--output_dir', output_dir])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(77)

    # load_model
    config_ = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config_)

    '''
        load dataset
            squad_kor_v1: Dataset Card for KorQuAD v1.0
    '''
    dataset = load_dataset('squad_kor_v1')
    MLM_train, MLM_eval = dataset['train'], dataset['validation']
    
    if do_eval:
        train_dataset = dataset['train'][:]['question']
        eval_dataset = dataset['validation'][:]['question']
        MLM_train = MLM_Dataset(
            queries = train_dataset,
            tokenizer = tokenizer,
            model_name = model_name,
            max_length = max_seq_length
        )
        MLM_eval = MLM_Dataset(
            queries = eval_dataset,
            tokenizer = tokenizer,
            model_name = model_name,
            max_length = max_seq_length
        )
    else:
        train_dataset = dataset['train'][:]['question'] + dataset['validation'][:]['question']
        MLM_train = MLM_Dataset(
            queries = train_dataset,
            tokenizer = tokenizer,
            model_name = model_name,
            max_length = max_seq_length
        )
    
    # load data collator
    if do_whole_word_mask:
        data_collator = DataCollatorForWholeWordMask(
            tokenizer = tokenizer,
            mlm = True,
            mlm_probability = masking_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = tokenizer,
            mlm = True,
            mlm_probability = masking_probability
        )

    # training args
    training_args.output_dir=f"{output_dir}/" + model_name.replace('/', '_')
    training_args.logging_dir = f"{training_args.output_dir}/logs"
    training_args.num_train_epochs = num_train_epochs
    training_args.learning_rate = learning_rate
    training_args.evaluation_strategy="steps" if MLM_eval is not None else "no"
    training_args.per_device_train_batch_size = per_device_train_batch_size
    training_args.per_device_eval_batch_size = per_device_eval_batch_size
    training_args.eval_steps = eval_steps
    training_args.save_steps = save_steps
    training_args.logging_steps = logging_steps
    training_args.warmup_steps = warmup_steps
    training_args.weight_decay = weight_decay
    training_args.load_best_model_at_end=True if MLM_eval is not None else False
    training_args.save_total_limit = save_total_limit

    training_args = TrainingArguments(
        output_dir = training_args.output_dir,
        logging_dir = training_args.logging_dir,
        num_train_epochs = training_args.num_train_epochs,
        learning_rate = training_args.learning_rate,
        evaluation_strategy = training_args.evaluation_strategy,
        per_device_train_batch_size = training_args.per_device_train_batch_size,
        per_device_eval_batch_size = training_args.per_device_eval_batch_size,
        eval_steps = training_args.eval_steps,
        save_steps = training_args.save_steps,
        logging_steps = training_args.logging_steps,
        warmup_steps = training_args.warmup_steps,
        weight_decay = training_args.weight_decay,
        load_best_model_at_end = training_args.load_best_model_at_end,
        save_total_limit = training_args.save_total_limit,
        run_name = model_name
    )
    print(training_args)

    # model to device
    model.to(device)

    # trainer 
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=MLM_train,
        eval_dataset=MLM_eval,
    )

    # train
    print("Start Training !!!")
    trainer.train()

    # model save
    print("Saving Training Result !!!")
    trainer.save_model()

    # finish training
    print("Finish Training !!!")


if __name__ == "__main__":
    # configuation
    cfg = OmegaConf.load(f'/opt/ml/level2_mrc_nlp-level2-nlp-01/code/conf/MLM/base_config.yaml')

    main(cfg)