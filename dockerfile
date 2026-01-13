# Base image with CUDA + Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install PostgreSQL and Python
RUN apt-get update && apt-get install -y \
    postgresql postgresql-contrib \
    python3 python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

#RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
#    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
#    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list \
#    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit \
#    sudo systemctl restart docker

# Create a PostgreSQL user & database
RUN service postgresql start && \
    su postgres -c "psql -c \"ALTER USER postgres WITH PASSWORD 'postgres123';\"" && \
    su postgres -c "psql -c \"CREATE USER julia WITH PASSWORD '123';\"" && \
    su postgres -c "psql -c \"CREATE DATABASE msft_customers OWNER julia;\""

# Copy requirements & Python script
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install pyairports from git to get latest version
RUN pip3 install git+https://github.com/ozeliger/pyairports.git

# Bake in model weights
# Usage: docker build --build-arg HF_TOKEN=$HF_TOKEN --build-arg MODEL_ID="..." -t my-image .
COPY constants.py .
COPY download_weights.py .
ARG HF_TOKEN
ARG MODEL_ID="none"
ARG MODEL_DIR="/app/models/baked_model"
# Set as env vars so python script picks them up
ENV MODEL_ID=$MODEL_ID
ENV MODEL_DIR=$MODEL_DIR
RUN python3 download_weights.py

# MARFT
COPY MARFT/ /app/MARFT/
RUN chmod +x /app/MARFT/marft/scripts/sample_redteam_script.sh

# Results directory for checkpoints
VOLUME /app/MARFT/marft/scripts/results

# PostgresDB
COPY data/ /app/data/
COPY schema.sql /app/schema.sql
COPY script/ ./script/
RUN chmod +x ./script/import_csvs.sh ./script/init.sh

# Access rules
COPY access_rules/ /app/access_rules/

# mcp
COPY mcp/ /app/mcp/
COPY util/ /app/util/

# agent
COPY run_model_and_agents.sh /app/run_model_and_agents.sh
RUN chmod +x /app/run_model_and_agents.sh
COPY host_models.py /app/host_models.py
COPY agent_loop.py .

# Set environment variables for Postgres
ENV PGHOST=localhost
ENV PGUSER=julia
ENV PGPASSWORD=123
ENV PGDATABASE=msft_customers

# Start Postgres, models, and run script

CMD /app/run_model_and_agents.sh
