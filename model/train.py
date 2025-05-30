import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
import os, tempfile                     # NEW

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

    # --- WandB ----------------------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.run_name,
            config=args,
        )
    # --------------------------------------------------------------------------

    # ---------- checkpoint bookkeeping ----------------------------------------
    tmp_dir = tempfile.gettempdir()
    current_ckpt_path = os.path.join(tmp_dir, f"{run_name}_current.pt")
    best_ckpt_path    = os.path.join(tmp_dir, f"{run_name}_best.pt")
    best_accuracy = float("-inf")              # RENAMED
    # --------------------------------------------------------------------------

    for phase in range(args.total_phases):
        # learning_phase now returns accuracy
        phase_accuracy = trainer.learning_phase()
        if phase_accuracy is None:             # fallback
            phase_accuracy = getattr(trainer, "latest_accuracy", 0.0)

        # -------- save & log current checkpoint -------------------------------
        torch.save(model.state_dict(), current_ckpt_path)
        if wandb_run:
            current_artifact = wandb.Artifact(f"{run_name}-current", type="model")
            current_artifact.add_file(current_ckpt_path)
            wandb_run.log_artifact(current_artifact, aliases=["current"])
        # ----------------------------------------------------------------------

        # -------- update best checkpoint --------------------------------------
        if phase_accuracy > best_accuracy:
            best_accuracy = phase_accuracy
            torch.save(model.state_dict(), best_ckpt_path)
            if wandb_run:
                best_artifact = wandb.Artifact(f"{run_name}-best", type="model")
                best_artifact.add_file(best_ckpt_path)
                wandb_run.log_artifact(best_artifact, aliases=["best"])
        # ----------------------------------------------------------------------
        # ...existing logging / metrics...

    # ------------------------ final model -------------------------------------
    final_ckpt_path = os.path.join(tmp_dir, f"{run_name}_final.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    if wandb_run:
        final_artifact = wandb.Artifact(f"{run_name}-final", type="model")
        final_artifact.add_file(final_ckpt_path)
        wandb_run.log_artifact(final_artifact, aliases=["final"])
    # --------------------------------------------------------------------------

    # close the single WandB run after all phases are done
    if wandb_run:                              # moved outside the loop
        wandb.finish()

    print("Training completed and model saved to WandB artifacts.")

if __name__ == "__main__":
    main()
