# Project Summary

More and more applications use machine learning to derive insights from large data collections. However, this process is susceptible to several security and privacy threats. For example, the data collection may contain sensitive, private information that may still be derived from the model or the learning and inference process. We work on several projects that help ensure that such threats are contained. We work on devising improved attacks that demonstrate that protection mechanisms are not as successful as they claim to be or processes that are assumed to be safe are not. We also work on defense mechanisms that provide better protection based on the latest developments in cryptography, differential privacy, and machine learning. Our work involves designing algorithms, developing prototypes, mostly in Python, and evaluating their performance and security.


## Set Up

Clone the repo:
```
git clone git@github.com:liya-zhu/ml-security.git
```

Add the database file to `/data/msft_customers.db`, or change the file_path at the beginning of `agent_loop.py` to wherever it is.


We are currently testing between two redteaming training libraries/approaches: Agent-lightning and MARFT.

## AgentLightning

Follow the instructions in the agentlightning/ repository to set it up. The attempt at redteaming is in agentlightning/examples/redteaming.

## MARFT

Follow the instructions in the MARFT/ repository to set it up. The attempt at redteaming can be run via MARFT/marft/scripts/sample_redteam_script.sh

## HuggingFace Token

To run models within the docker container that are gated behind a HuggingFace token, you can set up the `export HF_TOKEN="<token here>"` line in run_model_and_agents.sh and sample_redteam_script.sh.

## Running MARFT Dockerized

To get started, you can build the docker image and run with specific GPU IDs and viewing the logs from MARFT:
```
docker build -t test-image .
docker run -e HF_TOKEN --gpus '"device=6,7"' --name test-container test-image
docker logs -f test-container
```

If you want to debug the vLLM logs while the original container is running you can view the logs live using the following command:
```
docker exec -it test-container tail -f /app/model_server.log
```

If you want to stop and rebuild a container to fix an issue:
```
docker stop test-container
docker remove test-container
...<make your changes>...
docker build -t test-image .
```

### Important Files

dockerfile: Dockerized container to run MARFT in docker on growl

host_models.py: Sets up vLLM model services under separate processes. You can use the `setup_model_server` function to setup a model on a specific port.

util/mcp.py and mcp/postgres.py: MCP connector between the database and the database agent

script/...: Bash scripts run during the docker container initialization, sets up the database within the container.

access_rules/...: SQL files used to set up access control for the database
