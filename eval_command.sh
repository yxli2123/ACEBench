#!/usr/bin/env bash

#python -m vllm.entrypoints.openai.api_server \
#--served-model-name Qwen3-8B-FC \
#--model /ebs-basemodeling/liyixiao/models/Qwen2.5-7B-Instruct/ \
#--tensor-parallel-size 2 \
#--dtype bfloat16 \
#--max-model-len 32768 \
#--host 0.0.0.0 \
#--port 8181 \
#--gpu-memory-utilization 0.9 \
#--enable-auto-tool-choice \
#--tool-call-parser hermes

python generate.py \
--model_name Qwen3-8B-FC \
--user-model-name Qwen3-8B-FC \
--category multi_turn \
--num-threads 4 \
--api-url http://localhost:8181 \
--api-key EMPTY
