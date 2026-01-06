"""
Central configuration constants for model handling and Docker integration.
"""

# The default model to download/use if no overrides are provided
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Location where baked-in model weights are stored inside the container
DEFAULT_LOCAL_DIR = "/app/models/baked_model"

# Filename for saving build-time metadata about the model
BAKED_MODEL_INFO_FILE = "baked_model_info.json"

# Full path to the metadata file
BAKED_MODEL_INFO_PATH = f"/app/{BAKED_MODEL_INFO_FILE}"
