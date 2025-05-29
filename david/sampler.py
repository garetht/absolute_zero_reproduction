# %%
from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from jaxtyping import Int, Float
from typing import List, Tuple, Any

import constants
from model.args import AZRArgs


def generate_with_logprobs(
        args: AZRArgs,
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        prompts: List[str],
) -> Tuple[
    Int[torch.Tensor, "batch max_new_tokens"],
    Int[torch.Tensor, "batch max_new_tokens d_vocab"],
    Float[torch.Tensor, "batch max_new_tokens"]]:
    """
    Generate text completions for a batch of prompts and compute log probabilities.

    Args:
        model: The language model to use for generation
        prompts: List of input prompt strings
        message_format: Message format parameter (currently unused)
        max_new_tokens: Maximum number of new tokens to generate (default: 10)

    Returns:
        Tuple containing:
            - completion_ids: Tensor of shape (batch_size, max_new_tokens) with generated token IDs
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
    completion_ids = input_ids[:, prompt_len:]

    # logprobs_per_token = eindex(all_logprobs[:, :-1], completion_ids[:, 1:], "b s [b s] -> b s")
    logprobs_per_token = torch.gather(all_logprobs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
    print(f"{logprobs_per_token.shape=}")
    return completion_ids, all_logprobs, logprobs_per_token


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
