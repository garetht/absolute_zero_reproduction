MODEL_NAME=andrewzh/Absolute_Zero_Reasoner-Coder-3b

BENCHMARK=humaneval_plus
#BENCHMARK=hellaswag

export HF_ALLOW_CODE_EVAL=1

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME \
    --tasks $BENCHMARK \
    --device cuda:0 \
    --batch_size auto:64
    --include-unsafe # This flag is necessary to run hum