import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Int
from model.args import AZRArgs


def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input: str) -> Int[torch.Tensor, "seq"]:
    pass
