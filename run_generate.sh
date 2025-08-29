#!/usr/bin/env bash

# An example to run generation of FC mode
python -m vllm.entrypoints.openai.api_server \
--served-model-name Qwen2_5-14B-FC-Think \
--model /ebs-basemodeling/liyixiao/models/Qwen2.5-14B-Instruct/ \
--tensor-parallel-size 2 \
--dtype bfloat16 \
--max-model-len 32768 \
--host 0.0.0.0 \
--port 8181 \
--gpu-memory-utilization 0.9 \
--enable-auto-tool-choice \
--tool-call-parser hermes \
> vllm.log 2>&1 &

VLLM_PID=$!

# Wait for vllm to start
sleep 300

python generate.py \
--model_name Qwen2_5-14B-FC-Think \
--category test_all \
--language en \
--num-threads 4 \
--api-url http://localhost:8181/v1 \
--api-key EMPTY \
--user-model-name Qwen2_5-14B-FC-Think \
--user-api-url http://localhost:8181/v1 \
--user-api-key EMPTY \
--max-dialog-turns 32

# Kill the vllm and wait for cleaning
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null

# An example to run generation of prompt mode
python -m vllm.entrypoints.openai.api_server \
--served-model-name Qwen2_5-14B-Think \
--model /ebs-basemodeling/liyixiao/models/Qwen2.5-14B-Instruct/ \
--tensor-parallel-size 2 \
--dtype bfloat16 \
--max-model-len 32768 \
--host 0.0.0.0 \
--port 8181 \
--gpu-memory-utilization 0.9 \
> vllm.log 2>&1 &

VLLM_PID=$!

# Wait for vllm to start
sleep 300

python generate.py \
--model_name Qwen2_5-14B-Think \
--category test_all \
--language en \
--num-threads 4 \
--api-url http://localhost:8181/v1 \
--api-key EMPTY \
--user-model-name Qwen2_5-14B-Think \
--user-api-url http://localhost:8181/v1 \
--user-api-key EMPTY \
--max-dialog-turns 32

# Kill the vllm and wait for cleaning
kill $VLLM_PID
wait $VLLM_PID 2>/dev/null
