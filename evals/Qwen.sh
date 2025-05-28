MODEL_NAME=Qwen/Qwen2.5-Coder-3B
MODEL_NAME=Qwen/Qwen3-1.7B

BENCHMARK=hellaswag
BENCHMARK=tinyGSM8k

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks $BENCHMARK \
    --device cuda:0 \
    --batch_size auto:4