# RAG-FIRE

This repository contains the implementation of our method for processing data in the RAG pipeline. The models and datasets used in this project can be accessed from the provided link.

## Environment Setup

To create the environment, run the following commands:

```
conda create -n fire python=3.8
pip install -r requirement.txt
```


## Models and Data

The models and data used in this project can be downloaded from [this link](https://www.alipan.com/s/x4xbpUpscMw). The `models` folder contains the weights for our filter and polisher. The `datasets` folder includes the top 100 retrieved documents for NQ, TriviaQA, SQuAD, and WQ, as well as the data processed by our filter and polisher and some test files.

## Running the Method on RAG Pipeline

To evaluate the EM metric of the data processed by our method on the RAG pipeline, use the following command:

```
CUDA_VISIBLE_DEVICES=1 \
python eval_qa_rag_fire.py \
    --model_name Llama-3-8b \
    --dataset_path datasets/retriever/nq-test-npr-enhance.jsonl \
    --output_dir NQ \
    --num_docs 5 \
    --score_path datasets/filter/NQ_top5_filter.jsonl \
    --summary_path datasets/filter/NQ_top5_polisher.jsonl \
```


## Testing the Filter and Polisher

### Filter

To test our filter, use the following command:

```
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_filter.py \
    --input_path datasets/test/test_set_all.jsonl \
    --output_path datasets/test/test_set_scores_all.jsonl \
    --model_path models/filter \
```


### Polisher

To test our polisher, use the following command:

```
CUDA_VISIBLE_DEVICES=0 \
    python evaluate_polisher.py \
    --data_path datasets/test/ \
    --model_path models/polisher \
```
