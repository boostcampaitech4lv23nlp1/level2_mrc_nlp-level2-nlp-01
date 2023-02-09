# level2_mrc_nlp-level2-nlp-01
## Members
|강혜빈|권현정|백인진|이용우|이준원|
|:--:|:--:|:--:|:--:|:--:|
|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216918705-56e2f4d6-bc4f-482a-b9fd-190ca865d0e5.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216918785-3bc90fc4-e4b8-43f4-bd61-d797cf87e344.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919234-e9cc433c-f464-4a4b-8601-cffa668b22b2.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919814-f6ff7c2f-90ea-489c-b19a-a29fca8f9861.png">|<img width="100" alt="에브리타임" src="https://user-images.githubusercontent.com/37149278/216919925-1ab02487-e7a5-4995-8d22-1253bbcae550.png">|
|데이터 탐색,<br>코드 리뷰|모델링-Retriever, <br> Researcher|모델링-Reader, <br> Project Manager|모델링-Reader, <br> Researcher|모델링-Retriever, <br> 코드리뷰|
|[@hyeb](https://github.com/hyeb)|[@malinmalin2](https://github.com/malinmalin2)|[@eenzeenee](https://github.com/eenzeenee)|[@wooy0ng](https://github.com/wooy0ng)|[@jun9603](https://github.com/jun9603)


<br><br><br>

## Introduction
### 프로젝트 개요
Question Answering (QA)는 다양한 종류의 질문에 대답하는 인공지능을 만드는 연구 분야이다.

다양한 QA 시스템 중, Open-Domain Question Answering (ODQA)는 

주어진 지문이 별도로 존재하지 않고 사전 구축된 Knowledge Resource에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가된다.
 
이번 프로젝트에서는

- 관련 문서를 찾는 "Retriever"
- 관렴 문서를 읽고 적절한 답변을 찾거나 만드는 "Reader"

두가지 단계를 적절히 합쳐 질문에 답변을 할 수 있는 ODQA 시스템을 만드는 것이 목표이다.


<br><br><br>

## Wrap-up Report

[링크 참고](https://docs.google.com/document/d/18PV4IvGU4tZ9nFPE0dSJ6MKMfqqoL8MC1ylwCLnpDFo/edit?usp=sharing)


<br><br><br>

## 환경설정

Reader 환경 설정
```bash
./install/install_requirements.sh
```

<br>
Elastic Search 설치 및 사용 설정

```bash
./code/elastic_install.sh
./code/elastic_setting.py
```

<br><br><br>

## 하이퍼 파라미터 변경
하이퍼 파라미터는 code/conf 폴더 내 yaml 파일을 통해 변경이 가능하다.

```
|-- code
|   |-- conf
|   |   |-- MLM
|   |   |   `-- base_config.yaml
|   |   |-- reader
|   |   |   |-- base_config.yaml
|   |   |   |-- koelectra_config.yaml
|   |   |   |-- roberta-large_config.yaml
|   |   |   |-- roberta-large_config_39.17.yaml
|   |   |   |-- roberta-large_config_40.00.yaml
|   |   |   |-- roberta-large_config_51.25.yaml
|   |   |   `-- roberta-large_config_62.08.yaml
|   |   |-- retrieval
|   |   |   `-- base_config.yaml
|   |   `-- wandb_sweep
|   |       `-- wandb_config.yaml
```

<br><br>

### 아래 파일은 Reader 모델을 학습하기 위한 파라미터 설정이다.

```
# code/conf/reader/roberta-large_config.yaml

reader:
    mode:
        generation: False # 생성 기반 reader 모델을 활용할 것인지
train:
    path:
        output_dir: ./models/finetuning_dataset/ # 학습한 모델을 저장할 경로
        dataset_name: ../data/train_dataset/ # 학습 데이터셋 경로
        delete_exist_output: True # 현재 존재하는 모델 저장 경로일 경우 내부 내용을 삭제할지 여부

    stage:
        do_train: True
        do_eval: True
        overwrite_cache: True

    model:
        model_name_or_path: './models/pretraining_dataset'
        tokenizer_name: './models/pretraining_dataset'
        label_smoothing_factor: 0.0
        num_train_epochs: 3
        per_device_train_batch_size: 32
        learning_rate: 3e-5
        warmup_steps: 250
        weight_decay: 0.005
        fp16: True
test:
    path:
        output_dir: ./outputs/test_dataset/
        dataset_name: ../data/test_dataset/
        delete_exist_output: False

    stage:
        do_train: False
        do_eval: False
        do_predict: True

    model:
        model_name_or_path: ./models/finetuning_dataset/
```

<br><br>

### 아래 파일은 Retriever 모델을 학습하기 위한 파라미터 설정이다.

```
# code/conf/retrieval/base_config.yaml

path:
    dataset_name: ../data/train_dataset/
    data_path: ../data

model:
    model_name_or_path: 'klue/roberta-large'
    context_path: wikipedia_documents.json
    use_faiss: False
    bm25: True

```

<br><br>

### 아래 파일은 Retriever 모델에서 활용하는 파라미터 설정

```
# code/arguments.py

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    # line 79 부터
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=100,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )

    use_elastic: bool = field(
        default = True, metadata={"help" : "Whether to build with elastic search"},
    )
    
    elastic_index_name: str = field(
        default = "origin-wiki", metadata= {"help" : "Define the elastic search name"},
    )
```


<br><br>


### 아래 파일은 Reader 모델의 하이퍼 파라미터 변경 실험을 위한 설정이다.

```
# code/conf/wandb_sweep/wandb_sweep.yaml

method: 'random'
parameters:
    learning_rate:
        distribution: 'uniform'
        min: 1e-5
        max: 1e-4
    label_smoothing_factor:
        values: [0., 0.1]
    warmup_steps:
        values: [250, 500]
    weight_decay:
        values: [0.01, 0.005]
metric:
    name: 'train/loss'
    goal: 'minimize'
```



<br><br><br>


## Reference 

- DAPT / TAPT paper : [https://arxiv.org/abs/2004.10964](https://arxiv.org/abs/2004.10964)
- Dense Passage Retrieval : [https://arxiv.org/pdf/2004.04906v3.pdf](https://arxiv.org/pdf/2004.04906v3.pdf)
- BM25 + Cross Encoder : [https://arxiv.org/pdf/2104.08663v4.pdf](https://arxiv.org/pdf/2104.08663v4.pdf)
- Question Generation code : [https://github.com/codertimo/KorQuAD-Question-Generation](https://github.com/codertimo/KorQuAD-Question-Generation)




