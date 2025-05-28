import torch
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from constants import DEVICE
from model.args import AZRArgs


# returns the str response and the logprobs for the response
def generate_response(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        prompt: str,
) -> tuple[
    str,
    Float[torch.Tensor, "max_response_len vocab_size"],
    Int[torch.Tensor, "max_response_len"],
    Int[torch.Tensor, "prompt_len"],
    Int[torch.Tensor, "max_response_len"],
]:
    responses, logprobs, gen_ids, prompt_ids, attention_masks = generate_response_bulk(
        args, model, tokenizer, [prompt]
    )
    return responses[0], logprobs[0], gen_ids[0], prompt_ids[0], attention_masks[0]


# for each prompt in the list, returns a tuple of lists: (list of str responses, list of logprobs tensors)
# returns responses (list of strings), logprobs (of the gen responses), generated_ids (tokens of the generated response), inputs.input_ids (tokenized input prompts)
def generate_response_bulk(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        prompts: list[str],
) -> tuple[
    list[str],
    Float[torch.Tensor, "batch_size max_response_len vocab_size"],
    Int[torch.Tensor, "batch_size max_response_len"],
    Int[torch.Tensor, "batch_size prompt_len"],
    Int[torch.Tensor, "batch_size max_response_len"],
]:
    print("=" * 80)
    print("Preparing to call model with prompts: ")
    for prompt in prompts:
        print(prompt)
        print('-' * 20)

    # Tokenize inputs with padding
    inputs = tokenizer(
        prompts,
        padding=True,  # Pad to longest in batch
        truncation=True,
        return_tensors="pt",
    )

    # Generate responses
    outputs = model.generate(
        inputs.input_ids.to(DEVICE),
        attention_mask=inputs.attention_mask.to(DEVICE),
        max_new_tokens=args.max_response_length,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Extract generated tokens (excluding input tokens), shape (batch_size, actual_length)
    generated_ids = outputs.sequences[:, inputs.input_ids.shape[1]:]
    actual_length = generated_ids.shape[1]

    # Decode responses
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Pad generated_ids to max_response_length
    if actual_length < args.max_response_length:
        padding_length = args.max_response_length - actual_length
        padding = torch.full(
            (generated_ids.shape[0], padding_length),
            tokenizer.pad_token_id,
            dtype=generated_ids.dtype,
            device=generated_ids.device
        )
        generated_ids = torch.cat([generated_ids, padding], dim=1)
    elif actual_length > args.max_response_length:
        # Truncate if longer than expected
        generated_ids = generated_ids[:, :args.max_response_length]
        actual_length = args.max_response_length

    # Create attention masks to identify valid (non-padded) positions
    attention_masks = torch.zeros_like(generated_ids, dtype=torch.int)
    attention_masks[:, :actual_length] = 1

    # If we have EOS tokens, mask everything after the first EOS in each sequence
    if tokenizer.eos_token_id is not None:
        # Find first EOS position in each sequence (only in valid region)
        eos_positions = (generated_ids[:, :actual_length] == tokenizer.eos_token_id).int()
        if eos_positions.sum() > 0:
            # Create cumulative sum to mask everything after first EOS
            eos_cumsum = torch.cumsum(eos_positions, dim=1)
            # Mask positions after first EOS (but keep the EOS token itself)  
            post_eos_mask = (eos_cumsum <= 1).int()
            attention_masks[:, :actual_length] = attention_masks[:, :actual_length] * post_eos_mask

    # Process logits and pad to max_response_length
    # logits are these shape: (actual_length, batch_size, vocab_size) before transpose
    logits = torch.stack(outputs.scores, dim=0).transpose(0, 1)  # (batch_size, actual_length, vocab_size)
    logprobs = torch.log_softmax(logits, dim=-1)
    print("Receiving responses from model")
    for response in responses:
        print(response)
        print('-' * 40)
    # Pad logprobs to max_response_length if needed
    if actual_length < args.max_response_length:
        padding_length = args.max_response_length - actual_length
        logprobs_padding = torch.zeros(
            (logprobs.shape[0], padding_length, logprobs.shape[2]),
            dtype=logprobs.dtype,
            device=logprobs.device
        )
        logprobs = torch.cat([logprobs, logprobs_padding], dim=1)
    elif actual_length > args.max_response_length:
        logprobs = logprobs[:, :args.max_response_length, :]

    return responses, logprobs, generated_ids, inputs.input_ids, attention_masks


def compute_logprobs_for_tokens(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    prompts: list[str],
    tokens: Int[torch.Tensor, "batch_size max_response_len"],
    attention_masks: Int[torch.Tensor, "batch_size max_response_len"],
) -> Float[torch.Tensor, "batch_size max_response_len vocab_size"]:
    """
    Compute logprobs for given tokens with gradients enabled.
    This is used during training to get gradients through the policy.
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        tokens: Generated tokens from rollout phase
        attention_masks: Attention masks for valid positions
        
    Returns:
        logprobs with gradients enabled
    """
    # Tokenize prompts
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    
    # Create full sequence: prompt + generated tokens
    prompt_ids = inputs.input_ids.to(DEVICE)
    
    # Concatenate prompt and generated tokens
    full_sequence = torch.cat([prompt_ids, tokens], dim=1)
    
    # Create attention mask for full sequence
    prompt_attention = inputs.attention_mask.to(DEVICE)
    full_attention_mask = torch.cat([prompt_attention, attention_masks], dim=1)
    
    # Forward pass WITH gradients
    outputs = model(
        input_ids=full_sequence,
        attention_mask=full_attention_mask,
    )
    
    # Extract logits for the generated portion only
    logits = outputs.logits[:, prompt_ids.shape[1]:, :]  # Remove prompt portion
    
    # Ensure we have the right length
    if logits.shape[1] != tokens.shape[1]:
        # Pad or truncate to match expected length
        if logits.shape[1] < tokens.shape[1]:
            padding = torch.zeros(
                (logits.shape[0], tokens.shape[1] - logits.shape[1], logits.shape[2]),
                dtype=logits.dtype,
                device=logits.device
            )
            logits = torch.cat([logits, padding], dim=1)
        else:
            logits = logits[:, :tokens.shape[1], :]
    
    # Convert to log probabilities
    logprobs = torch.log_softmax(logits, dim=-1)
    
    return logprobs


def remove_dvocab_from_logprobs(
        logprobs: Float[torch.Tensor, "role task batch_size max_response_len vocab_size"],
        tokens: Int[torch.Tensor, "role task batch_size max_response_len"],
) -> Float[torch.Tensor, "role task batch_size max_response_len"]:
    """
    Extract log probabilities for generated tokens.

    Args:
        logprobs: Log probabilities for all tokens in the vocabulary
        tokens: Generated token IDs (already excluding prompt)

    Returns:
        Log probabilities for each generated token
    """

    # Extract log probabilities for the actual generated tokens
    # Use gather to select the log prob for each generated token
    return torch.gather(logprobs, dim=-1, index=tokens.unsqueeze(-1).to(torch.int64)
                        ).squeeze(-1)
