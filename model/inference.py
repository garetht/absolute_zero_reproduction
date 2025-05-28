import torch
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast 

from model.args import AZRArgs

# returns the str response and the logprobs for the response
def generate_response(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, prompt: str) -> tuple[str, Float[torch.Tensor, "max_response_len vocab_size"], Int[torch.Tensor, "max_response_len"], Int[torch.Tensor, "prompt_len"]]:
    responses, logprobs, gen_ids, prompt_ids = generate_response_bulk(args, model, tokenizer, [prompt])
    return responses[0], logprobs[0], gen_ids[0], prompt_ids[0]

# for each prompt in the list, returns a tuple of lists: (list of str responses, list of logprobs tensors)
# returns responses (list of strings), logprobs (of the gen responses), generated_ids (tokens of the generated response), inputs.input_ids (tokenized input prompts)
def generate_response_bulk(args: AZRArgs, model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, prompts: list[str]) -> tuple[list[str], Float[torch.Tensor, "batch_size max_response_len vocab_size"], Int[torch.Tensor, "batch_size max_response_len"], Int[torch.Tensor, "batch_size prompt_len"]]:
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
    
    # Extract generated tokens (excluding input tokens), shape (batch_size, max_response_length)
    generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
    
    # Decode responses
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # TODO confirm that we aren't off by 1 indexing into the logits here
    # logits are these shape: (max_response_length, batch_size, vocab_size_size) before transpose, so transpose to (batch_size, max_response_length, vocab_size_size)
    logits = torch.stack(outputs.scores, dim=0).transpose(0, 1)
    logprobs = torch.log_softmax(logits, dim=-1)
    # logprobs shape: (batch_size, max_response_length vocab_size_size)
    return responses, logprobs, generated_ids, inputs.input_ids

def remove_dvocab_from_logprobs(logprobs: Float[torch.Tensor, "batch_size max_response_len vocab_size"], 
                 tokens: Int[torch.Tensor, "batch_size max_response_len"]) -> Float[torch.Tensor, "batch_size max_response_len"]:
    """
    Extract log probabilities for generated tokens.
    
    Args:
        logprobs: Log probabilities for all tokens in the vocab_sizeulary
        tokens: Generated token IDs (already excluding prompt)
        
    Returns:
        Log probabilities for each generated token
    """
    
    # Extract log probabilities for the actual generated tokens
    # Use gather to select the log prob for each generated token
    return torch.gather(
        logprobs,
        dim=-1,
        index=tokens.unsqueeze(-1)
    ).squeeze(-1)