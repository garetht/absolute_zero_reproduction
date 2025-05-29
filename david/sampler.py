# %%
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from jaxtyping import Int, Float
from typing import List, Tuple, Any

import constants
from model.args import AZRArgs


def generate_with_logprobs_2(model: AutoModelForCausalLM,
                             tokenizer: AutoTokenizer,
                             prompts: List[str],
                             args: AZRArgs
                             ) -> Tuple[Int[torch.Tensor, "batch max_new_tokens"],
Float[torch.Tensor, "batch max_new_tokens d_vocab"], Int[torch.Tensor, "batch max_new_tokens"], Int[
    torch.Tensor, "batch max_new_tokens"]]:
    """
    Generate text completions for a batch of prompts and compute log probabilities.

    Args:
        model: The language model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        prompts: List of input prompt strings
        args: AZRArgs object containing generation parameters (max_response_length, top_p, temperature)

    Returns:
        Tuple containing:
            - completion_ids: Tensor of shape (batch_size, max_new_tokens) with generated token IDs
            - attention_mask: Tensor of shape (batch_size, max_new_tokens) with attention mask up to EOS token
            - logprobs_per_token: Tensor of shape (batch_size, max_new_tokens) with log probabilities
    """

    inputs = tokenizer(prompts,
                       return_tensors="pt",
                       padding=True,
                       padding_side="left")

    prompt_len = inputs.input_ids.shape[1]

    inputs = inputs.to(constants.DEVICE)

    input_ids = model.generate(**inputs,
                               max_new_tokens=args.max_response_length,
                               use_cache=True,
                               do_sample=True,
                               top_p=args.top_p,
                               temperature=args.temperature)

    # note this may consume a lot of memory
    # might need to do this in chunks
    logits = model(input_ids).logits[:, prompt_len - 1:-1]
    logprobs = torch.log_softmax(logits, dim=-1)
    completion_ids = input_ids[:, prompt_len:]

    # logprobs_per_token = eindex(logprobs[:, :-1], completion_ids[:, 1:], "b s [b s] -> b s")
    logprobs_per_token = torch.gather(logprobs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    eos_mask = (ids == tokenizer.eos_token_id)
    eos_positions = torch.argmax(eos_mask.int(), dim=-1)
    # Handle cases where no EOS token is found (argmax returns 0 even if no match)
    no_eos_mask = ~eos_mask.any(dim=-1)
    eos_positions[no_eos_mask] = ids.shape[1]  # Use sequence length if no EOS found

    # Generate attention mask that pays attention to everything up to EOS token (vectorized)
    batch_size, seq_len = ids.shape
    position_indices = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = (position_indices <= eos_positions.unsqueeze(1)).int()

    return completion_ids, logprobs, logprobs_per_token, attention_mask


def generate_with_logprobs(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        prompts: List[str],
) -> Tuple[
    Int[torch.Tensor, "batch max_new_tokens"],
    Int[torch.Tensor, "batch max_new_tokens d_vocab"],
    Float[torch.Tensor, "batch max_new_tokens"],
    Int[torch.Tensor, "batch max_new_tokens"],

]:
    """
    Generate text completions for a batch of prompts and compute log probabilities.

    Args:
        model: The language model to use for generation
        prompts: List of input prompt strings
        message_format: Message format parameter (currently unused)
        max_new_tokens: Maximum number of new tokens to generate (default: 10)

    Returns:
        Tuple containing:
            - generated_ids: Tensor of shape (batch_size, max_new_tokens) with generated token IDs
            - logprobs_per_token: Tensor of shape (batch_size, max_new_tokens) with log probabilities
    """

    inputs = tokenizer(prompts,
                       return_tensors="pt",
                       padding=True,
                       padding_side="left")
    prompt_len = inputs.input_ids.shape[1]
    inputs = inputs.to(constants.DEVICE)

    input_ids = model.generate(**inputs,
                               max_new_tokens=args.max_response_length,
                               use_cache=True,
                               do_sample=True,
                               top_p=args.rollout_top_p,
                               temperature=args.rollout_temperature)

    # note this may consume a lot of memory
    # might need to do this in chunks
    logits = model(input_ids).logits[:, prompt_len - 1:-1]
    all_logprobs = torch.log_softmax(logits, dim=-1)
    generated_ids = input_ids[:, prompt_len:]

    actual_length = generated_ids.shape[1]

    # logprobs_per_token = eindex(all_logprobs[:, :-1], generated_ids[:, 1:], "b s [b s] -> b s")
    logprobs_per_token = torch.gather(all_logprobs, dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)

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

    # Pad logprobs to max_response_length if needed
    if actual_length < args.max_response_length:
        padding_length = args.max_response_length - actual_length
        logprobs_padding = torch.zeros(
            (all_logprobs.shape[0], padding_length, all_logprobs.shape[2]),
            dtype=all_logprobs.dtype,
            device=all_logprobs.device
        )
        all_logprobs = torch.cat([all_logprobs, logprobs_padding], dim=1)
    elif actual_length > args.max_response_length:
        all_logprobs = all_logprobs[:, :args.max_response_length, :]

    if actual_length < args.max_response_length:
        padding_length = args.max_response_length - actual_length
        logprobs_padding = torch.zeros(
            (logprobs_per_token.shape[0], padding_length),
            dtype=logprobs_per_token.dtype,
            device=logprobs_per_token.device
        )
        logprobs_per_token = torch.cat([logprobs_per_token, logprobs_padding], dim=1)
    elif actual_length > args.max_response_length:
        logprobs_per_token = logprobs_per_token[:, :args.max_response_length, :]

    print(f"{logprobs_per_token.shape=}")
    print(f"{all_logprobs.shape=}")
    return generated_ids, all_logprobs, logprobs_per_token, attention_masks


if __name__ == "__main__":

    tokenizer = GPT2Tokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # torch.compile(model)

    prompts = [
        "How does the Qwen2.5-1.5B-Instruct model work?",
        "What is the difference between the Qwen2.5-1.5B-Instruct and the Qwen2.5-0.5B model?",
        "Who is Sam Altman?",
    ]

    # Convert each prompt to chat format
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(text)

    ids, logprobs = generate_with_logprobs(model, formatted_prompts, max_new_tokens=50)
    completions = tokenizer.batch_decode(ids, skip_special_tokens=True)
    for formatted_prompt, completion, logprob in zip(formatted_prompts, completions, logprobs):
        print(formatted_prompt + completion)
        print(logprob.sum().item())
        print("------")

# %%
