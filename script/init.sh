# start postgres
service postgresql start && \
    psql -U julia -d msft_customers -f /app/schema.sql && \
    PGUSER=julia PGDATABASE=msft_customers /app/script/import_csvs.sh && \
    echo "DEBUG: Checking tables..." && \
    PGPASSWORD=postgres123 psql -U postgres -d msft_customers -c "SELECT schemaname, tablename FROM pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema');" && \
    PGPASSWORD=postgres123 psql -U postgres -d msft_customers -f /app/access_rules/row_security_setup.sql

# start mcp server
python3 mcp/postgres.py &