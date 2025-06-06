from datetime import datetime
import torch
import os, tempfile
import wandb                               # NEW
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file   # NEW
from tqdm import tqdm
from buffer.base_buff import MegaBuffer
from constants import CHECKPOINT_DIR, MODEL_NAME, DEVICE
from custom_types import Role, TaskType
from model.args import AZRArgs
from model.trainer import AZRTrainer
from utils.mocks.mock_transformer import MockAutoModelForCausalLM
import gc  # Import garbage collector


# ---------------------------------------------------------------------------
# Helper to robustly save checkpoints (avoids zip-writer bug & GPU tensors)
def save_model_checkpoint(model, path: str):
    """Save model.state_dict() on CPU in the .safetensors format."""
    state_dict_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, path)      # safetensors write
# ---------------------------------------------------------------------------


def main():
    wandb_project_name = "AZR"
    use_mock = False
    run_name = f"AZR-Run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
 
    # -------- WandB run ------------------------------------------------------
    wandb_run = None
    if args.use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=args.run_name,
            config=args,
        )
    # -------------------------------------------------------------------------

    # --- local checkpoint directory -------------------------------------------
    ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # --------------------------------------------------------------------------

    tmp_dir = tempfile.gettempdir()        # still used for temp saving
    current_ckpt_path = os.path.join(tmp_dir,  f"{run_name}_current.safetensors")
    best_ckpt_path    = os.path.join(ckpt_dir, f"{run_name}_best.safetensors")

    # ---------- checkpoint bookkeeping ----------------------------------------
    best_accuracy = float("-inf")
    # --------------------------------------------------------------------------
    epoch = 0
    for phase in tqdm(range(args.total_phases), desc=f"Epoch {epoch+1}/{args.total_phases}"):
        epoch += 1
        
        # Clear CUDA cache at the start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        phase_accuracy = trainer.learning_phase() or getattr(trainer,
                                                             "latest_accuracy",
                                                             0.0)
        if wandb_run:                       # log metrics
            wandb_run.log({"phase": phase, "accuracy": phase_accuracy})

        # # -------- save current checkpoint (temp) ------------------------------
        # with torch.no_grad():  # Ensure no gradients during checkpoint save
        #     save_model_checkpoint(model, current_ckpt_path)
        #  # ----------------------------------------------------------------------

        # # -------- update best checkpoint --------------------------------------
        # if phase_accuracy > best_accuracy:
        #     best_accuracy = phase_accuracy
        #     with torch.no_grad():  # Ensure no gradients during checkpoint save
        #         save_model_checkpoint(model, best_ckpt_path)
         # ----------------------------------------------------------------------
        # ...existing logging / metrics...
        
        # Clear cache after checkpoint saving
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection

    # ------------------------ summary -----------------------------------------
    # print(f"Best checkpoint saved to: {best_ckpt_path}")
    # --------------------------------------------------------------------------

    if wandb_run:                           # close run
        wandb_run.finish()

    print("Training completed and best model checkpoint saved locally.")

if __name__ == "__main__":
    main()
