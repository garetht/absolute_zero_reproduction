from transformers import AutoModelForCausalLM, AutoTokenizer

from model.args import AZRArgs

# returns the str response and the logprobs for the response
def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> tuple[str, Float[torch.Tensor, "seq_len vocab"]]:
    responses, logprobs = generate_response_bulk(args, model, tokenizer, [prompt])
    return responses[0], logprobs[0]

# for each prompt in the list, returns a tuple of lists: (list of str responses, list of logprobs tensors)
def generate_response_bulk(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: list[str]) -> tuple[list[str], Float[torch.Tensor, "batch_size seq_len vocab"]]:
    pass
