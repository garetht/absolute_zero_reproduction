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
# Helper to robustly save checkpoints with accelerator support
def save_model_checkpoint(accelerator: Accelerator, model, tokenizer, path: str, is_best: bool = False):
    """Save model, tokenizer, and training state using accelerator."""
    if accelerator.is_main_process:
        # Save the model and tokenizer
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        
        # Save additional metadata
        metadata = {
            "is_best": is_best,
            "timestamp": datetime.now().isoformat(),
        }
        
        import json
        with open(os.path.join(path, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Saved {'best ' if is_best else ''}checkpoint to {path}")

def push_to_hub_if_configured(accelerator: Accelerator, model, tokenizer, repo_name: str, commit_message: str):
    """Push model to HuggingFace Hub if configured."""
    if not accelerator.is_main_process:
        return
        
    try:
        from huggingface_hub import HfApi
        
        # Check if we can access the hub (will raise if not logged in)
        api = HfApi()
        user_info = api.whoami()
        
        if user_info:
            print(f"🚀 Pushing model to HuggingFace Hub as {repo_name}...")
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.push_to_hub(
                repo_name,
                commit_message=commit_message,
                private=False,  # Set to True if you want private repos
            )
            tokenizer.push_to_hub(
                repo_name,
                commit_message=commit_message,
            )
            print(f"✅ Successfully pushed to https://huggingface.co/{user_info['name']}/{repo_name}")
        
    except ImportError:
        print("⚠️  huggingface_hub not installed. Skipping push to hub.")
    except Exception as e:
        print(f"⚠️  Could not push to hub: {e}")
        print("   Make sure you're logged in with: huggingface-cli login")
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
        accelerator=accelerator,  # Pass accelerator to MegaBuffer
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

    # --- checkpoint configuration ---------------------------------------------
    save_checkpoints = os.path.exists(CHECKPOINT_DIR)
    if save_checkpoints and accelerator.is_main_process:
        ckpt_dir = os.path.join(CHECKPOINT_DIR, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        current_ckpt_path = os.path.join(ckpt_dir, "current")
        best_ckpt_path = os.path.join(ckpt_dir, "best")
        print(f"📁 Checkpoints will be saved to: {ckpt_dir}")
    elif accelerator.is_main_process:
        print(f"⚠️  Checkpoint directory {CHECKPOINT_DIR} does not exist. Skipping checkpoint saving.")
    
    # ---------- checkpoint bookkeeping ----------------------------------------
    best_accuracy = float("-inf")
    has_saved_checkpoints = False
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

        # -------- save checkpoints if enabled ----------------------------------
        if save_checkpoints:
            # Always save current checkpoint
            with torch.no_grad():
                save_model_checkpoint(accelerator, model, tokenizer, current_ckpt_path, is_best=False)
                has_saved_checkpoints = True
            
            # Save best checkpoint if this is the best accuracy so far
            if phase_accuracy > best_accuracy:
                best_accuracy = phase_accuracy
                with torch.no_grad():
                    save_model_checkpoint(accelerator, model, tokenizer, best_ckpt_path, is_best=True)
                
                if accelerator.is_main_process:
                    print(f"🏆 New best accuracy: {best_accuracy:.2%}")
        # ----------------------------------------------------------------------
        
        # Clear cache after checkpoint saving
        if accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()  # Force garbage collection

    # ------------------------ post-training actions --------------------------
    if save_checkpoints and has_saved_checkpoints and accelerator.is_main_process:
        print(f"📊 Training completed! Best accuracy: {best_accuracy:.2%}")
        print(f"💾 Current checkpoint: {current_ckpt_path}")
        print(f"🏆 Best checkpoint: {best_ckpt_path}")
        
        # Push best model to HuggingFace Hub if enabled
        if args.push_to_hub:
            repo_name = f"{args.hub_repo_prefix}-{run_name.lower()}"
            commit_message = f"AZR model trained for {args.total_phases} phases, best accuracy: {best_accuracy:.2%}"
            
            try:
                # Load the best checkpoint before pushing
                print("📤 Loading best checkpoint for Hub upload...")
                best_model = AutoModelForCausalLM.from_pretrained(best_ckpt_path)
                best_tokenizer = AutoTokenizer.from_pretrained(best_ckpt_path)
                
                push_to_hub_if_configured(accelerator, best_model, best_tokenizer, repo_name, commit_message)
            except Exception as e:
                print(f"⚠️  Could not load best checkpoint for Hub upload: {e}")
        else:
            print("🔒 Hub upload disabled in config")
    
    elif accelerator.is_main_process:
        print("Training completed (no checkpoints saved).")
    # --------------------------------------------------------------------------

    if wandb_run and accelerator.is_main_process:  # close run only on main process
        wandb_run.finish()

if __name__ == "__main__":
    main()
