import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

from buffer.base_buff import MegaBuffer
from constants import MODEL_NAME, DEVICE
from custom_types import Role, TaskType
from model.args import AZRArgs
from model.trainer import AZRTrainer
from utils.mocks.mock_transformer import MockAutoModelForCausalLM


def main():
    wandb_project_name = "AZR"
    use_mock = False
    run_name = "AZR-Run"

    if use_mock:
        model = MockAutoModelForCausalLM()
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map=DEVICE)

    args = AZRArgs(
        wandb_project_name=wandb_project_name,
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
        lr=args.lr,  # do we want to set beta?
    )


    mega_buffer = MegaBuffer(
        args=args,
        logprobs=torch.empty(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
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

    # --- checkpoint bookkeeping ---
    ckpt_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_reward = float("-inf")
    best_ckpt_path = None
    prev_ckpt_path = None
    # ------------------------------

    for phase in range(args.total_phases):
        phase_reward = trainer.learning_phase()  # assumes the method returns the reward
        if phase_reward is None:                 # fallback for trainers that don't return it
            phase_reward = getattr(trainer, "latest_reward", 0.0)

        # ---- save current checkpoint ----
        curr_ckpt_path = os.path.join(ckpt_dir, f"{args.run_name}_phase_{phase}.pt")
        torch.save(model.state_dict(), curr_ckpt_path)

        # keep only two checkpoints: best and current
        if prev_ckpt_path and prev_ckpt_path != best_ckpt_path:
            # remove the superseded “current” checkpoint
            try:
                os.remove(prev_ckpt_path)
            except OSError:
                pass
        prev_ckpt_path = curr_ckpt_path
        # ---------------------------------

        # ---- update best checkpoint ----
        if phase_reward > best_reward:
            # remove the previous best (unless it is the same as curr_ckpt_path)
            if best_ckpt_path and best_ckpt_path != curr_ckpt_path:
                try:
                    os.remove(best_ckpt_path)
                except OSError:
                    pass
            best_reward = phase_reward
            best_ckpt_path = curr_ckpt_path
        # ---------------------------------

        # Add any additional logging here
        # ...existing code...

    if args.use_wandb:
        wandb.finish()

    # Save the final model
    final_model_path = os.path.join(ckpt_dir, f"{args.run_name}_final.pt")
    torch.save(model.state_dict(), final_model_path)

    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
