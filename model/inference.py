from transformers import AutoModelForCausalLM, AutoTokenizer

from model.args import AZRArgs


def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    return generate_response_bulk(args, model, tokenizer, [prompt])[0]


def generate_response_bulk(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: list[str]) -> list[str]:
    pass
