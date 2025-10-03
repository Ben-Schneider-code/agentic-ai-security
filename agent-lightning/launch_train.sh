# python sql_agent.py \
#       --litsqlagent.trained-agents write \
#       --trainer.n-workers 1 \
#       --trainer.dev true  # Enable the dev debug mode.

export VERL_API_BASE=http://localhost:9997/  # Same as the server port. This is used for receiving tasks and sending results.
python sql_agent.py \
      --litsqlagent.trained-agents write \
      --trainer.n-workers 16 \
      --litsqlagent.val-temperature 0