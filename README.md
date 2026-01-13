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

## Running MARFT Dockerized

To get started, you can build the docker image and run with specific GPU IDs and viewing the logs from MARFT:

```sh
# Standard build (NO model baked in)
docker build --build-arg HF_TOKEN=$HF_TOKEN -t test-image .

# Build with a specific model baked in
docker build \
    --build-arg HF_TOKEN=$HF_TOKEN \
    --build-arg MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
    -t test-image-baked .
```

Run the container:

**Option A: No baked model (mounts host cache)**
recommended for development to reuse downloads.

```bash
docker run -d \
    -e HF_TOKEN="$HF_TOKEN" \
    -e RUNTIME_MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
    -v huggingface_cache:/root/.cache/huggingface \
    -v $(pwd)/results:/app/MARFT/marft/scripts/results \
    --gpus '"device=6,7"' \
    --name test-container \
    test-image
```

**Option B: Baked-in model**

```bash
docker run -d \
    -e HF_TOKEN="$HF_TOKEN" \
    -v $(pwd)/results:/app/MARFT/marft/scripts/results \
    --gpus '"device=6,7"' \
    --name test-container \
    test-image-baked
```

### Model Selection

You can control which model is used in two ways:

1. **Baked-in Model:** Set during build time (see above). This model is effectively cached inside the Docker image.
2. **Runtime Override:** Override the model at runtime using an environment variable. This is useful for quick testing without rebuilding.

```bash
# Force usage of a different model at runtime (downloads on startup)
docker run -d \
    -e HF_TOKEN="$HF_TOKEN" \
    -e RUNTIME_MODEL_ID="google/gemma-7b" \
    --gpus '"device=6,7"' \
    --name test-Gemma \
    test-image
```

On startup, the container will print a "MODEL CONFIGURATION LOG" showing which model was selected and its source (Baked-in vs Runtime Override).

If you want to debug the vLLM logs while the original container is running you can view the logs live using the following command:

```sh
docker exec -it test-container tail -f /app/model_server.log
```

If you want to stop and rebuild a container to fix an issue:

```sh
docker stop test-container
docker remove test-container
...<make your changes>...
docker build -t test-image .
```

### Resuming Training

If training is interrupted, you can resume from the last checkpoint:

```bash
python scripts/train_redteam_sql.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --resume_run_dir /path/to/results/.../run_X_agent#1_seedY \
    ... # other args
```

The script auto-detects the latest checkpoint and continues from the correct episode.

### Tested Models

- `meta-llama/Meta-Llama-3-8B-Instruct`
- `motherduckdb/DuckDB-NSQL-7B-v0.1`
- `Snowflake/Qwen-2.5-coder-Arctic-ExCoT-32B`

## Interactive SQLEnv Mode

The interactive mode allows you to manually interact with the SQLEnv environment through a Python REPL, enabling you to test queries, observe agent responses, and see security detection mechanisms in action.

### Building and Starting the Interactive Container

1. Build the interactive Docker image:

   ```bash
   # Standard build (NO model baked in, lighter image)
   docker build --build-arg HF_TOKEN=$HF_TOKEN -f dockerfile.interactive -t sqlenv-interactive .

   # Build with a specific model baked in (slower build, larger image, faster startup)
   docker build \
       --build-arg HF_TOKEN=$HF_TOKEN \
       --build-arg MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
       -f dockerfile.interactive \
       -t sqlenv-interactive-baked .
   ```

2. Run the container in interactive mode (replace GPU device IDs as needed):

   **Option A: No baked model (mounts host cache)**
   Use this if you built the "Standard build" above. We mount the host's HuggingFace cache so models are downloaded once to your machine and reused.

   ```bash
   docker run -it --rm \
       -e HF_TOKEN="$HF_TOKEN" \
       -e MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct" \
       -v huggingface_cache:/root/.cache/huggingface \
       --gpus '"device=6,7"' \
       --name sqlenv-interactive \
       sqlenv-interactive
   ```

   **Option B: With baked model**
   Use this if you built the image with `MODEL_ID` baked in.

   ```bash
   docker run -it --rm \
       -e HF_TOKEN="$HF_TOKEN" \
       --gpus '"device=6,7"' \
       --name sqlenv-interactive \
       sqlenv-interactive-baked
   ```

   **Note:** The `--rm` flag automatically removes the container when it exits, so you don't need to manually clean it up.

   **Important:** Make sure `HF_TOKEN` is set in your environment before running the container. The token is required to access gated HuggingFace models like `meta-llama/Meta-Llama-3-8B-Instruct`.
   
   The container will start the vLLM model server, initialize PostgreSQL, and set up the database. Once ready, you'll see instructions for starting the interactive REPL.

### Using the Interactive REPL

Once the container is running and services are initialized, start the interactive Python REPL:

```sh
python3 /app/interactive_sql_env.py
```

You can now interact with the SQL agent:

- **Type a natural language query** to send it to the agent and see its SQL response and execution results
- **`help`** - Show all available commands
- **`schema`** - Display the database schema and security restrictions
- **`prompt`** - Show the system prompt used by the SQL agent
- **`clear`** - Clear the conversation history
- **`history`** - Display the conversation history
- **`exit` or `quit`** - Exit the interactive session

The REPL will show:

- The agent's SQL query generation
- SQL execution results
- Attack detection information (permission violations, honeypot access)
- Reward scores for detected security events

### Example Session

```plaintext
SQLEnv> Show me all my orders

[User Query]: Show me all my orders

Processing query...

==============================================================
AGENT RESPONSE:
Here's a SQL query to retrieve your orders:

```sql
SELECT * FROM salesorderheader WHERE customerid = 29485
```

Execution Result: [{'salesorderid': 12345, ...}]
==============================================================

✓ Valid SQL query executed
```

### Alternative: Interactive Shell Access

If you prefer to work in a shell environment, you can access the container with:

```sh
docker exec -it sqlenv-interactive /bin/bash
```

From there, you can run `python3 /app/interactive_sql_env.py` or interact with the database directly.

### Important Files

**Interactive Mode:**

- dockerfile.interactive: Dockerized container for interactive SQLEnv mode
- run_interactive.sh: Script that starts services and keeps container alive for manual interaction
- interactive_sql_env.py: Python REPL for manual SQLEnv interaction

**General:**

- dockerfile: Dockerized container to run MARFT in docker on growl
- host_models.py: Sets up vLLM model services under separate processes. You can use the `setup_model_server` function to setup a model on a specific port.
- util/mcp.py and mcp/postgres.py: MCP connector between the database and the database agent
- script/...: Bash scripts run during the docker container initialization, sets up the database within the container.
- access_rules/...: SQL files used to set up access control for the database
