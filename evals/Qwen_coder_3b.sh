MODEL_NAME=Qwen/Qwen2.5-Coder-3B

BENCHMARK=humaneval_plus
BENCHMARK=hellaswag

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks $BENCHMARK \
    --device cuda:0 \
    --batch_size auto:4