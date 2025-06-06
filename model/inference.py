import torch
from jaxtyping import Float, Int
from transformers import AutoModelForCausalLM, BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from typing import Tuple

from constants import DEVICE
from model.args import AZRArgs



def generate_without_grads(model: AutoModelForCausalLM, inputs: BatchEncoding, tokenizer: PreTrainedTokenizerFast, max_new_tokens: int, device: torch.device) -> \
        tuple[Int[torch.Tensor, "batch_size max_response_len"], Float[torch.Tensor, "batch_size max_response_len"]]:
    """
    Generate text completions without computing gradients.
    
    Args:
        model: The language model to use for generation
        inputs: Tokenized input batch with input_ids and attention_mask
        tokenizer: Tokenizer for the model
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to run generation on
        
    Returns:
        Tuple containing:
            - generated_ids: Shape (batch_size, max_response_len) - Generated token IDs (excluding input tokens)
            - logprobs_per_token: Shape (batch_size, max_response_len) - Log probabilities for each generated token
    """


    outputs = model.generate(
        inputs.input_ids.to(DEVICE),
        attention_mask=inputs.attention_mask.to(DEVICE),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        use_cache=True,
    )
    generated_ids = outputs.sequences[:, inputs.input_ids.shape[1] :]


    scores = torch.stack(outputs.scores, dim=1)
    logprobs = torch.log_softmax(scores, dim=-1)
    
    logprobs_per_token = logprobs.gather(
        dim=-1,
        index=generated_ids.unsqueeze(-1)
    ).squeeze(-1)

    return generated_ids, logprobs_per_token



@torch.no_grad()
def generate_response_bulk(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        prompts: list[str],
) -> tuple[
    list[str],
    Float[torch.Tensor, "batch_size max_response_len"],
    Int[torch.Tensor, "batch_size max_response_len"],
    Int[torch.Tensor, "batch_size prompt_len"],
    Int[torch.Tensor, "batch_size max_response_len"],
]:
    """
    Generate text completions for a batch of prompts without computing gradients.
    
    Args:
        args: Configuration arguments containing max_prompt_length and max_response_length
        model: The language model to use for generation
        tokenizer: Tokenizer for the model
        prompts: List of input prompt strings
        
    Returns:
        Tuple containing:
            - responses: List of decoded response strings
            - logprobs: Shape (batch_size, max_response_len) - Log probabilities for generated tokens
            - generated_ids: Shape (batch_size, max_response_len) - Generated token IDs (padded/truncated)
            - input_ids: Shape (batch_size, prompt_len) - Tokenized input prompts
            - attention_masks: Shape (batch_size, max_response_len) - Masks for valid (non-padded) positions
    """




    # Tokenize inputs with padding
    inputs = tokenizer(
        prompts,
        padding='max_length',  # Pad to longest in batch
        max_length=args.max_prompt_length,
        truncation=True,
        return_tensors="pt",
    )

    # Generate responses
    generated_ids, logprobs = generate_without_grads(
        model, inputs, tokenizer, args.max_response_length, DEVICE
    )

    # Extract generated tokens (excluding input tokens), shape (batch_size, actual_length)
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

    # Pad logprobs to max_response_length if needed
    if actual_length < args.max_response_length:
        padding_length = args.max_response_length - actual_length
        logprobs_padding = torch.zeros(
            (logprobs.shape[0], padding_length),
            dtype=logprobs.dtype,
            device=logprobs.device
        )
        logprobs = torch.cat([logprobs, logprobs_padding], dim=1)
    elif actual_length > args.max_response_length:
        logprobs = logprobs[:, :args.max_response_length, :]

    return responses, logprobs, generated_ids, inputs.input_ids, attention_masks


def generate_with_logprobs(model: AutoModelForCausalLM,
                           tokenizer: PreTrainedTokenizerFast,
                           prompts: list[str],
                           args: AZRArgs,
                           debug: bool = False
                           ) -> Tuple[
    Int[torch.Tensor, "batch max_new_tokens"],
    Int[torch.Tensor, "batch max_new_tokens"],
    Float[torch.Tensor, "batch max_new_tokens"]
]:
    """
    Generate text completions for a batch of prompts and compute log probabilities.

    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        prompts: List of input prompt strings
        args: AZRArgs object containing generation parameters (max_response_length, rollout_top_p, rollout_temperature)
        debug: Whether to print debug information

    Returns:
        Tuple containing:
            - completion_ids: Shape (batch, max_response_length) - Generated token IDs (padded/truncated)
            - attention_mask: Shape (batch, max_response_length) - Attention mask up to EOS token
            - logprobs_per_token: Shape (batch, max_response_length) - Log probabilities for generated tokens
    """

    inputs = tokenizer(prompts,
                       return_tensors="pt",
                       padding=True,
                       padding_side="left")

    prompt_len = inputs.input_ids.shape[1]

    inputs = inputs.to(DEVICE)

    input_ids = model.generate(**inputs,
                               max_new_tokens=args.max_response_length,
                               use_cache=True,
                               do_sample=True,
                               top_p=args.rollout_top_p,
                               temperature=args.rollout_temperature)

    logits = model(input_ids).logits

    if debug:
        print(f"{logits.shape=}")
        print(f"{prompt_len=}")

    logits = logits[:, prompt_len - 1:-1]

    if debug:
        print(f"{logits.shape=}")

    logprobs = torch.log_softmax(logits, dim=-1)
    completion_ids = input_ids[:, prompt_len:]

    logprobs_per_token = torch.gather(logprobs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    eos_mask = (completion_ids == tokenizer.eos_token_id)
    eos_positions = torch.argmax(eos_mask.int(), dim=-1)
    # Handle cases where no EOS token is found (argmax returns 0 even if no match)
    no_eos_mask = ~eos_mask.any(dim=-1)
    eos_positions[no_eos_mask] = completion_ids.shape[1]  # Use sequence length if no EOS found

    # Generate attention mask that pays attention to everything up to EOS token (vectorized)
    batch_size, response_len = completion_ids.shape
    position_indices = torch.arange(response_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = (position_indices <= eos_positions.unsqueeze(1)).int()

    # Pad/truncate to max_response_length for consistent tensor shapes
    target_length = args.max_response_length
    current_length = logprobs_per_token.size(1)

    if debug:
        print(f"before : {logprobs_per_token.shape=}")
        print(f"before : {attention_mask.shape=}")

    if current_length < target_length:
        # Pad with zeros
        pad_size = target_length - current_length
        logprobs_per_token = torch.nn.functional.pad(logprobs_per_token, (0, pad_size), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size), value=0)
        completion_ids = torch.nn.functional.pad(completion_ids, (0, pad_size), value=0)

    elif current_length > target_length:
        raise ValueError(f"Generated sequence is longer than the target length: {current_length} > {target_length}")

    assert logprobs_per_token.shape == attention_mask.shape == completion_ids.shape, f"inference.py: {logprobs_per_token.shape=} {attention_mask.shape=} {completion_ids.shape=}"

    if debug:
        print(f"{logprobs_per_token.shape=}")
        print(f"{attention_mask.shape=}")

    return completion_ids, attention_mask, logprobs_per_token


