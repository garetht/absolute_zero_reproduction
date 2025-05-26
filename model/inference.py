from transformers import AutoModelForCausalLM, AutoTokenizer


def generate_response(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args: AZRArgs, input: str) -> str:
    pass