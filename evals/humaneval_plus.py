from lm_eval import evaluator
from pprint import pprint

MODEL_ID = "andrewzh/Absolute_Zero_Reasoner-Coder-3b"  # this is TOO BIG on a 16G GPU
MODEL_ID = "Qwen/Qwen3-0.6B"  # STILL TOO BIG for a 16G GPU
MODEL_ID = "EleutherAI/pythia-160m"  # barely fits in GPU memory

results = evaluator.simple_evaluate(
    model="hf",
    model_args="pretrained=" + MODEL_ID,
    tasks=["humaneval_plus"],
    batch_size=64,
    device="cuda:0",
    confirm_run_unsafe_code=True,
)

pprint(results["results"])
