# %%
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from jaxtyping import Int, Float
from typing import List, Tuple, Any

import constants
from model.args import AZRArgs




def generate_with_logprobs_2(model: AutoModelForCausalLM,
                             tokenizer: PreTrainedTokenizerFast,
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
                               top_p=args.rollout_top_p,
                               temperature=args.rollout_temperature)

    # note this may consume a lot of memory
    # might need to do this in chunks
    logits = model(input_ids).logits


    logits = logits[:, prompt_len - 1:-1]


    logprobs = torch.log_softmax(logits, dim=-1)
    completion_ids = input_ids[:, prompt_len:]

    # logprobs_per_token = eindex(logprobs[:, :-1], completion_ids[:, 1:], "b s [b s] -> b s")
    logprobs_per_token = torch.gather(logprobs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    eos_mask = (completion_ids == tokenizer.eos_token_id)
    eos_positions = torch.argmax(eos_mask.int(), dim=-1)
    # Handle cases where no EOS token is found (argmax returns 0 even if no match)
    no_eos_mask = ~eos_mask.any(dim=-1)
    eos_positions[no_eos_mask] = completion_ids.shape[1]  # Use sequence length if no EOS found

    # Generate attention mask that pays attention to everything up to EOS token (vectorized)
    batch_size, seq_len = completion_ids.shape
    position_indices = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = (position_indices <= eos_positions.unsqueeze(1)).int()

    # Pad/truncate to max_response_length for consistent tensor shapes
    target_length = args.max_response_length
    current_length = logprobs_per_token.size(1)


    if current_length < target_length:
        # Pad with zeros
        pad_size = target_length - current_length
        logprobs = torch.nn.functional.pad(logprobs, (0, 0, 0, pad_size), value=0)
        logprobs_per_token = torch.nn.functional.pad(logprobs_per_token, (0, pad_size), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_size), value=0)
        completion_ids = torch.nn.functional.pad(completion_ids, (0, pad_size), value=0)
    elif current_length > target_length:
        # Truncate to target length
        logprobs = logprobs[:, :target_length]
        attention_mask = attention_mask[:, :target_length]
        completion_ids = completion_ids[:, :target_length]


    return completion_ids, logprobs, logprobs_per_token, attention_mask


def generate_with_logprobs(model: AutoModelForCausalLM,
                           tokenizer: PreTrainedTokenizerFast,
                           prompts: List[str],
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
                               top_p=args.rollout_top_p,
                               temperature=args.rollout_temperature)

    # note this may consume a lot of memory
    # might need to do this in chunks
    logits = model(input_ids).logits

    if debug:
        print(f"{logits.shape=}")
        print(f"{prompt_len=}")

    logits = logits[:, prompt_len - 1:-1]

    if debug:
        print(f"{logits.shape=}")

    logprobs = torch.log_softmax(logits, dim=-1)
    completion_ids = input_ids[:, prompt_len:]

    # logprobs_per_token = eindex(logprobs[:, :-1], completion_ids[:, 1:], "b s [b s] -> b s")
    logprobs_per_token = torch.gather(logprobs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)

    eos_mask = (completion_ids == tokenizer.eos_token_id)
    eos_positions = torch.argmax(eos_mask.int(), dim=-1)
    # Handle cases where no EOS token is found (argmax returns 0 even if no match)
    no_eos_mask = ~eos_mask.any(dim=-1)
    eos_positions[no_eos_mask] = completion_ids.shape[1]  # Use sequence length if no EOS found

    # Generate attention mask that pays attention to everything up to EOS token (vectorized)
    batch_size, seq_len = completion_ids.shape
    position_indices = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0).expand(batch_size, -1)
    attention_mask = (position_indices <= eos_positions.unsqueeze(1)).int()

    # Pad/truncate to max_response_length for consistent tensor shapes
    target_length = args.max_response_length
    current_length = logprobs_per_token.size(1)

    if debug:
        print(f"before : {logprobs.shape=}")
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

    assert logprobs_per_token.shape == attention_mask.shape == completion_ids.shape, f"sampler.py: {logprobs_per_token.shape=} {attention_mask.shape=} {completion_ids.shape=}"

    if debug:
        print(f"{logprobs_per_token.shape=}")
        print(f"{attention_mask.shape=}")

    return completion_ids, attention_mask, logprobs_per_token


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
