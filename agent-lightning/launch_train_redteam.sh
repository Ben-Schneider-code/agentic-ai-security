export VERL_API_BASE=http://localhost:9998/  # Same as the server port. This is used for receiving tasks and sending results.
python red_sql_agent.py \
      --trainer.api_base http://localhost:9998 \
      --litadversarialsqlagent.trained_agents red_agent \
      --trainer.n_workers 16