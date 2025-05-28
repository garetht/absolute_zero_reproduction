MODEL_NAME=andrewzh/Absolute_Zero_Reasoner-Coder-3b
BENCHMARK=hellaswag

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks $BENCHMARK \
    --device cuda:0 \
    --batch_size auto:64