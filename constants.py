import os
import json

# The default model to download/use if no overrides are provided
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Location where baked-in model weights are stored inside the container
DEFAULT_LOCAL_DIR = "/app/models/baked_model"

# Filename for saving build-time metadata about the model
BAKED_MODEL_INFO_FILE = "baked_model_info.json"

# Full path to the metadata file
BAKED_MODEL_INFO_PATH = f"/app/{BAKED_MODEL_INFO_FILE}"


def get_runtime_model_id() -> tuple[str, str]:
    """
    Determine which model ID to use at runtime.
    Returns a tuple (model_id, source_description).
    """
    # 1. Check if user wants to override via env var (Highest priority)
    env_model_id = os.environ.get("RUNTIME_MODEL_ID") or os.environ.get("MODEL_ID")
    if env_model_id:
        return env_model_id, "Environment Variable (RUNTIME_MODEL_ID/MODEL_ID)"

    # 2. Check for baked-in model metadata
    if os.path.exists(BAKED_MODEL_INFO_PATH):
        try:
            with open(BAKED_MODEL_INFO_PATH, "r") as f:
                info = json.load(f)
                baked_path = info.get("local_dir")
                baked_id = info.get("model_id", "Unknown")

                # Verify the path actually exists
                if baked_path and os.path.exists(baked_path):
                    return baked_path, f"Baked-in Weights ({baked_id})"
        except Exception:
            pass  # Fallthrough if failed to read

    # 3. Default fallback
    return DEFAULT_MODEL_ID, "Hardcoded Default"
