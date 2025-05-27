import torch
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast 

from model.args import AZRArgs

# returns the str response and the logprobs for the response
def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, prompt: str) -> tuple[str, Float[torch.Tensor, "seq_len vocab"], Int[torch.Tensor, "seq_len"], Int[torch.Tensor, "prompt_len"]]:
    responses, logprobs, gen_ids, prompt_ids = generate_response_bulk(args, model, tokenizer, [prompt])
    return responses[0], logprobs[0], gen_ids[0], prompt_ids[0]

# for each prompt in the list, returns a tuple of lists: (list of str responses, list of logprobs tensors)
# returns responses (list of strings), logprobs (of the gen responses), generated_ids (tokens of the generated response), inputs.input_ids (tokenized input prompts)
def generate_response_bulk(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, prompts: list[str]) -> tuple[list[str], Float[torch.Tensor, "batch_size seq_len vocab"], Int[torch.Tensor, "batch_size seq_len"], Int[torch.Tensor, "batch_size prompt_len"]]:
    # TODO move these to where we instantiate the tokenizer
    # Set up tokenizer for left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize inputs with padding
    inputs = tokenizer(
        prompts,
        padding=True,  # Pad to longest in batch
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_response_length,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Extract generated tokens (excluding input tokens)
    generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
    
    # Decode responses
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Convert scores to logprobs
    # Stack scores: (max_response_length, batch_size, vocab_size)
    scores = torch.stack(outputs.scores, dim=0)
    # Transpose to (batch_size, max_response_length, vocab_size)
    logprobs = scores.transpose(0, 1).log_softmax(dim=-1)
    
    return responses, logprobs, generated_ids, inputs.input_ids