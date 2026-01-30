# Project Summary

More and more applications use machine learning to derive insights from large data collections. However, this process is susceptible to several security and privacy threats. For example, the data collection may contain sensitive, private information that may still be derived from the model or the learning and inference process. We work on several projects that help ensure that such threats are contained. We work on devising improved attacks that demonstrate that protection mechanisms are not as successful as they claim to be or processes that are assumed to be safe are not. We also work on defense mechanisms that provide better protection based on the latest developments in cryptography, differential privacy, and machine learning. Our work involves designing algorithms, developing prototypes, mostly in Python, and evaluating their performance and security.

## Set Up

Clone the repo:

```sh
git clone https://github.com/Ben-Schneider-code/agentic-ai-security.git
```

Add the database file to `/data/msft_customers.db`, or change the file_path at the beginning of `agent_loop.py` to wherever it is.

We are currently testing between two redteaming training libraries/approaches: Agent-lightning and MARFT.

## AgentLightning

Follow the instructions in the agentlightning/ repository to set it up. The attempt at redteaming is in agentlightning/examples/redteaming.

## MARFT

Follow the instructions in the MARFT/ repository to set it up. The attempt at redteaming can be run via MARFT/marft/scripts/sample_redteam_script.sh

## HuggingFace Token

To run models within the docker container that are gated behind a HuggingFace token, you can set an env variable using `export HF_TOKEN="<token here>"`. The token will be passed into the docker container.

## Docker Setup

The project uses a base image to share common dependencies across different environments.

### 1. Build Base Image (Required First)

You **must** build the base image before building any of the specific environment images.

```bash
docker build -f dockerfile.base -t agentic-security-base .
```

### 2. Build Specific Environment Images

Once the base image is built, you can build the image for your specific needs:

**A. Production (Training)**
Runs the full training loop with MARFT.

```bash
docker build -f dockerfile -t agentic-security-prod .
```

**B. Interactive Mode**
Allows manual interaction with the SQL environment via a REPL.

```bash
docker build -f dockerfile.interactive -t agentic-security-interactive .
```

**C. Testing Mode**
Starts the services (vLLM, Postgres) and drops you into a bash shell execution of custom scripts.

```bash
docker build -f dockerfile.testing -t agentic-security-testing .
```

## Running the Containers

### Running Testing Mode (Batch Conversations)

This mode is useful for running a list of pre-defined red team prompts (e.g., successful jailbreaks) against the blue team agent.

1. **Run the container** (starts vLLM & Postgres in background, then opens shell):

    ```bash
    docker run --gpus '"device=6,7"' -it \
        -e HF_TOKEN="$HF_TOKEN" \
        -v $(pwd)/conversations.txt:/app/conversations.txt \
        agentic-security-testing
    ```

    *Note: Mount your conversation file if it's not already in the image.*

2. **Run the conversation driver**:
    Inside the container shell:

    ```bash
    python3 /app/util/run_conversations.py /app/conversations.txt
    ```

    The script supports input files with blocks of text separated by blank lines. Lines starting with `#` are ignored.

### Running Interactive Mode

1. **Run the container**:

    ```bash
    docker run --gpus '"device=6,7"' -it --rm \
        -e HF_TOKEN="$HF_TOKEN" \
        agentic-security-interactive
    ```

2. **Start the REPL** (once inside):

    ```bash
    python3 /app/interactive_sql_env.py
    ```

### Running Production (MARFT Training)

```bash
docker run --gpus '"device=6,7"' -d \
    -e HF_TOKEN="$HF_TOKEN" \
    --name marft-training \
    agentic-security-prod
```

Logs can be viewed at `/app/redteam_output.log` or via `docker logs`.

## AgentLightning

Follow the instructions in the agentlightning/ repository to set it up. The attempt at redteaming is in agentlightning/examples/redteaming.

## MARFT

Follow the instructions in the MARFT/ repository to set it up. The attempt at redteaming can be run via MARFT/marft/scripts/sample_redteam_script.sh

## Important Files

- `dockerfile.base`: Base image with common dependencies (CUDA, Python, Postgres, vLLM).
- `dockerfile`: Production image for training.
- `dockerfile.interactive`: Interactive image with `interactive_sql_env.py`.
- `dockerfile.testing`: Testing image with `start_services_and_shell.sh`.
- `util/run_conversations.py`: Script to run batch conversations from a file.
- `start_services_and_shell.sh`: Startup script for the testing image.

**Interactive Mode:**

- dockerfile.interactive: Dockerized container for interactive SQLEnv mode
- run_interactive.sh: Script that starts services and keeps container alive for manual interaction
- interactive_sql_env.py: Python REPL for manual SQLEnv interaction
