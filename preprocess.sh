#!/bin/bash

python tools/preprocess_data.py \
       --input ../pyggy/data/pretrain/c4_sample.jsonl \
       --output-prefix gpt2c4 \
       --vocab-file gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod \
       --workers 8 \
       --chunk-size 512
