"""privprune: prompt-conditioned, privacy-aware visual token pruning for VLMs."""
import os

# Model weights for this project live on the second disk (the root disk is full).
# Must be set before transformers / huggingface_hub are imported; an existing HF_HOME wins.
os.environ.setdefault("HF_HOME", "/mnt/disk1/ahmed/hf")
