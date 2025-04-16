#!/bin/bash
# Script to restart and troubleshoot MariaDB container

# Stop and remove the current MariaDB container
echo "Stopping and removing the current MariaDB container..."
docker stop regulaite-mariadb 2>/dev/null
docker rm regulaite-mariadb 2>/dev/null

# Remove the MariaDB volume to start fresh
echo "Removing MariaDB volume..."
docker volume rm regulaite_mariadb_data 2>/dev/null

# Pull the latest MariaDB image
echo "Pulling the latest MariaDB image..."
docker pull mariadb:lts

# Create the network if it doesn't exist
if ! docker network inspect regulaite_network >/dev/null 2>&1; then
    echo "Creating regulaite_network..."
    docker network create regulaite_network
fi

# Start only the MariaDB container with docker-compose
echo "Starting the MariaDB container..."
docker-compose up -d mariadb

# Wait for the container to start
echo "Waiting for the MariaDB container to start..."
sleep 10

# Check if the container is running
if docker ps | grep -q regulaite-mariadb; then
    echo "MariaDB container is running."
    
    # Check logs
    echo "Container logs:"
    docker logs regulaite-mariadb

    # Check container info
    echo "Container info:"
    docker inspect regulaite-mariadb | grep -i entrypoint
    docker inspect regulaite-mariadb | grep -i cmd
    
    # Try to connect to MariaDB
    echo "Attempting to connect to MariaDB..."
    if docker exec -it regulaite-mariadb mariadb --user=regulaite_user --password=SecureP@ssw0rd! -e "SHOW DATABASES;"; then
        echo "Successfully connected to MariaDB."
    else
        echo "Failed to connect to MariaDB."
    fi
else
    echo "MariaDB container failed to start."
    echo "Container logs:"
    docker logs regulaite-mariadb
fi

echo "Script completed." 