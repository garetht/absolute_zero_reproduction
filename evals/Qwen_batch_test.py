from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

prompt_prefix = """

"""


def generate_prompts(num_prompts: int) -> list[str]:
    """
    Generate a list of prompts for testing.
    """
    prompts = []
    for _ in range(num_prompts):
        prime = np.random.choice()
        input = np.random.choice()
        prompts.append(
            f"{prompt_prefix}. What is the inverse of {input} modulo {prime}?"
        )
    return prompts


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, device_map="auto"
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity in simple terms.",
    ]

    # Set up tokenizer for left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Tokenize inputs with padding
    inputs = tokenizer(
        prompts,
        padding=True,  # Pad to longest in batch
        truncation=True,
        return_tensors="pt",
    ).to(model.device)  # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=32,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )  # Extract generated tokens (excluding input tokens)

    generated_ids = outputs.sequences[
        :, inputs.input_ids.shape[1] :
    ]  # Decode responses
    responses = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )  # Convert scores to logprobs
    # Stack scores: (max_response_length, batch_size, vocab_size)
    scores = torch.stack(outputs.scores, dim=0)
    # Transpose to (batch_size, max_response_length, vocab_size)
    logprobs = scores.transpose(0, 1).log_softmax(dim=-1)

    # print(f"{logprobs.shape = }")
    for question, response in zip(prompts, responses):
        print(f"Question: {question}")
        print(f"Response: {response}\n")
