# Start from base
# Build base first: docker build -f dockerfile.base -t agentic-security-base .
FROM agentic-security-base

# Add finetuning script
COPY run_model_and_agents.sh /app/run_model_and_agents.sh
RUN chmod +x /app/run_model_and_agents.sh

# Agent specific files
COPY experiment_label.txt /app/experiment_label.txt

# Start Postgres, models, and run script
CMD /app/run_model_and_agents.sh
