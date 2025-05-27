from transformers import AutoModelForCausalLM, AutoTokenizer

from model.args import AZRArgs

# returns the str response and the logprobs for the response
def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> tuple[str, Float[torch.Tensor, "vocab"]]:
    return generate_response_bulk(args, model, tokenizer, [prompt])[0]

# for each prompt in the list, returns a tuple of the str response and the logprobs for the response
def generate_response_bulk(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: list[str]) -> list[tuple[str, Float[torch.Tensor, "vocab"]]]:
    pass
