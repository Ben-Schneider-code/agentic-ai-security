# Base image with CUDA + Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install PostgreSQL and Python
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

#RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
#    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
#    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit \
#    sudo systemctl restart docker


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
COPY connect_agent_to_db.py /app/connect_agent_to_db.py
COPY host_models.py /app/host_models.py
COPY run_model_and_agents.sh /app/run_model_and_agents.sh
RUN chmod +x /app/import_csvs.sh
RUN chmod +x /app/run_model_and_agents.sh

# Set environment variables for Postgres
ENV PGHOST=localhost
ENV PGUSER=julia
ENV PGPASSWORD=123
ENV PGDATABASE=msft_customers

# Expose PostgreSQL port
EXPOSE 5432

# Expose ports for the two models for testing
EXPOSE 8000
EXPOSE 8001

# Start Postgres, models, and run script
CMD ["/app/run_model_and_agents.sh"]