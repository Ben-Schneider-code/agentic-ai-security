# Base image with CUDA + Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install PostgreSQL and Python
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Create a PostgreSQL user & database
RUN service postgresql start && \
    su postgres -c "psql -c \"CREATE USER julia WITH PASSWORD '123';\"" && \
    su postgres -c "psql -c \"CREATE DATABASE msft_customers OWNER julia;\""

# Copy requirements & Python script
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY agent_loop.py .
COPY data/ /app/data/
COPY schema.sql /app/schema.sql
COPY import_csvs.sh /app/import_csvs.sh
RUN chmod +x /app/import_csvs.sh

# Set environment variables for Postgres
ENV PGHOST=localhost
ENV PGUSER=julia
ENV PGPASSWORD=123
ENV PGDATABASE=msft_customers

# Expose PostgreSQL port
EXPOSE 5432

# Start Postgres and run script
CMD service postgresql start && \
    psql -U julia -d msft_customers -f /app/schema.sql && \
    /app/import_csvs.sh && \
    python3 agent_loop.py
