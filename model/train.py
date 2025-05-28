from custom_types import Role, TaskType
from model.args import AZRArgs
from model.trainer import AZRTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import wandb

from constants import MODEL_NAME, DEVICE
from custom_types import Role, TaskType

from buffer.base_buff import MegaBuffer
from utils.mocks.mock_transformer import MockAutoModelForCausalLM


def main():
    wandb_project_name = "AZR"
    use_wandb = False
    use_mock = False
    run_name = "AZR-Run"

    if use_mock:
        model = MockAutoModelForCausalLM()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE)
    args = AZRArgs(
        wandb_project_name=wandb_project_name,
        use_wandb=use_wandb,
        run_name=run_name,
        d_vocab=model.config.vocab_size
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",
    )
    # Set up tokenizer eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,  # do we want to set beta?,
        maximize=True,  # We want to maximize the reward, not minimize the loss
    )


    mega_buffer = MegaBuffer(
        args=args,
        logprobs=torch.empty(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
                args.d_vocab,
            ),
            device=DEVICE,
            dtype=args.dtype,
        ),
        sample_ids=torch.empty(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
            ),
            dtype=torch.int,
            device=DEVICE,
        ),
        attention_masks=torch.ones(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
            ),
            dtype=torch.int,
            device=DEVICE,
        ),
    )
    mega_buffer.initialize_seed_buffer(tokenizer)

    trainer = AZRTrainer(
        args=args,
        mega_buffer=mega_buffer,
        tokenizer=tokenizer,
        training_model=model,
        optimizer=optimizer,
        run_name=args.run_name,
    )

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.run_name,
            config=args,
        )

    for phase in range(args.total_phases):
        trainer.learning_phase()
        # Add logging here

    if args.use_wandb:
        wandb.finish()

    # Save the model

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
