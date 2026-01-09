#!/bin/bash
set -e

# Wait for PG to be ready (although initdb runs before it's fully "ready" for external connections, 
# for init scripts it acts as the superuser or the specified POSTGRES_USER)

echo "Keep going... initializing database..."

# 1. Schema
echo "Running schema.sql..."
psql -U julia -d msft_customers -f /app/schema.sql

# 2. Data Import
# The original script uses /app/script/import_csvs.sh. 
# We need to make sure permissions are right or just run the logic here.
echo "Running import_csvs.sh..."
chmod +x /app/script/import_csvs.sh
# We need to set PGPASSWORD because import script might not have it from env automatically if users switch
export PGPASSWORD=123
export PGUSER=julia
export PGDATABASE=msft_customers
/app/script/import_csvs.sh

# 3. Agent User & Row Security
# This script usually needs to run as a superuser (postgres)? 
# In this container, POSTGRES_USER=julia became the superuser/owner roughly.
# But `row_security_setup.sql` creates roles, which requires CREATEROLE privilege.
# 'julia' might need that privilege if it doesn't have it. 
# The official postgres image makes the POSTGRES_USER a superuser. So 'julia' should be able to create roles.

echo "Running row_security_setup.sql..."
psql -U julia -d msft_customers -f /app/access_rules/row_security_setup.sql

echo "Database initialization complete."
