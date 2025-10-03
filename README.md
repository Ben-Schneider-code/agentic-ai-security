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