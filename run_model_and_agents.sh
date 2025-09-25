#!/bin/bash
# Start the server in the background
python3 host_models.py 2>&1 | tee server.log &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Run other scripts
service postgresql start && \
    psql -U julia -d msft_customers -f /app/schema.sql && \
    /app/import_csvs.sh && \
    python3 connect_agent_to_db.py

# Keep container alive by bringing server to foreground
cat server.log

# Keep showing server logs in real-time
tail -f server.log &

# Keep container alive
wait $SERVER_PID