# start postgres
service postgresql start && \
    psql -U julia -d msft_customers -f /app/schema.sql && \
    /app/script/import_csvs.sh

# start mcp server
python3 mcp/postgres.py &