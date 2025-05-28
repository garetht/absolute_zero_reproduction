import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LOGGING_DIR = "/logs"
CHECKPOINT_DIR = "/checkpoints"
DATA_DIR = "/data"
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"





