from datetime import datetime
import torch
import os, tempfile
import wandb
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file
from tqdm import tqdm
from buffer.base_buff import MegaBuffer
from constants import CHECKPOINT_DIR, MODEL_NAME
from custom_types import Role, TaskType
from model.args import AZRArgs
from model.trainer import AZRTrainer
from utils.mocks.mock_transformer import MockAutoModelForCausalLM
import gc


# ---------------------------------------------------------------------------
# Helper to robustly save checkpoints (avoids zip-writer bug & GPU tensors)
def save_model_checkpoint(model, path: str):
    """Save model.state_dict() on CPU in the .safetensors format."""
    state_dict_cpu = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    save_file(state_dict_cpu, path)      # safetensors write
# ---------------------------------------------------------------------------


def main():
    # Initialize Accelerator first
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",  # Use bfloat16 for better performance
        log_with="wandb" if True else None,  # Will be configured later
    )
    
    wandb_project_name = "AZR"
    use_mock = False
    run_name = f"AZR-Run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Load model without device_map when using Accelerator
    if use_mock:
        model = MockAutoModelForCausalLM()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        )

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
        lr=args.lr,
    )

    # Prepare model and optimizer with Accelerator
    model, optimizer = accelerator.prepare(model, optimizer)

    # Create buffers on accelerator device
    device = accelerator.device
    mega_buffer = MegaBuffer(
        args=args,
        logprobs=torch.empty(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
            ),
            device=device,
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
            device=device,
        ),
        attention_masks=torch.ones(
            (
                len(Role),
                len(TaskType),
                args.batch_size,
                args.max_response_length,
            ),
            dtype=torch.int,
            device=device,
        ),
    )
    mega_buffer.initialize_seed_buffer(tokenizer)

    trainer = AZRTrainer(
        args=args,
        mega_buffer=mega_buffer,
        tokenizer=tokenizer,
        training_model=model,
        optimizer=optimizer,
        accelerator=accelerator,
        run_name=args.run_name,
    )
 
    # -------- WandB run ------------------------------------------------------
    wandb_run = None
    if args.use_wandb and accelerator.is_main_process:
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
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        
        phase_accuracy = trainer.learning_phase() or getattr(trainer,
                                                             "latest_accuracy",
                                                             0.0)
        if wandb_run and accelerator.is_main_process:  # log metrics only on main process
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
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection

    # ------------------------ summary -----------------------------------------
    # print(f"Best checkpoint saved to: {best_ckpt_path}")
    # --------------------------------------------------------------------------

    if wandb_run and accelerator.is_main_process:  # close run only on main process
        wandb_run.finish()

    if accelerator.is_main_process:
        print("Training completed and best model checkpoint saved locally.")

if __name__ == "__main__":
    main()
