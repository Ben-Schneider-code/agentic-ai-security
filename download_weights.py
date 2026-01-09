import json
import os
from huggingface_hub import snapshot_download
import constants


def download_model(model_id: str, local_dir: str):
    print(f"Downloading model {model_id} to {local_dir}...")

    # Ensure token is present
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN environment variable not set. Download of gated models (like Llama 3) will fail."
        )

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=[
            "*.pt",
            "*.bin",
        ],  # Prefer safetensors if available, optional optimization
    )
    print(f"Successfully downloaded {model_id}")

    # Save metadata for runtime
    info = {"model_id": model_id, "local_dir": local_dir}
    with open(constants.BAKED_MODEL_INFO_FILE, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved build info to {constants.BAKED_MODEL_INFO_FILE}")


if __name__ == "__main__":
    # Allow overriding via env vars or just use defaults
    # If MODEL_ID is set to empty string or "none", we skip downloading
    model_id = os.environ.get("MODEL_ID", constants.DEFAULT_MODEL_ID)
    local_dir = os.environ.get("MODEL_DIR", constants.DEFAULT_LOCAL_DIR)

    if not model_id or model_id.lower() == "none":
        print("MODEL_ID set to 'none' or empty. Skipping model download.")
        # We do NOT write the baked_model_info.json file, so runtime will know nothing is baked.
    else:
        download_model(model_id, local_dir)
