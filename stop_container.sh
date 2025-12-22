#!/bin/bash
# Script to stop and remove a Docker container by name

if [ $# -eq 0 ]; then
    echo "Usage: $0 <container_name>"
    echo "Example: $0 sqlenv-interactive"
    exit 1
fi

CONTAINER_NAME=$1

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '${CONTAINER_NAME}' does not exist."
    exit 1
fi

# Check if container is running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping container '${CONTAINER_NAME}'..."
    docker stop "${CONTAINER_NAME}"
    if [ $? -eq 0 ]; then
        echo "✓ Container stopped successfully"
    else
        echo "✗ Failed to stop container"
        exit 1
    fi
else
    echo "Container '${CONTAINER_NAME}' is not running (already stopped)"
fi

# Remove the container
echo "Removing container '${CONTAINER_NAME}'..."
docker rm "${CONTAINER_NAME}"
if [ $? -eq 0 ]; then
    echo "✓ Container removed successfully"
else
    echo "✗ Failed to remove container"
    exit 1
fi

echo "Done!"



