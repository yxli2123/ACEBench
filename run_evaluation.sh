#!/usr/bin/env bash

python evaluate.py \
--model-name Qwen2_5-14B-FC-Think \
--language en \
--category test_all

python evaluate.py \
--model-name Qwen2_5-14B-Think \
--language en \
--category test_all
