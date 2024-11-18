#!/bin/bash

############################################################
# Prerequisites:
# 1. Build SageMath 10.0+ from source
# 2. Adjust docker run paths according to your environment
############################################################

# Define container name
CONTAINER_NAME="transformer-gb"

# Check existing container
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Container ${CONTAINER_NAME} already exists."
    
    # Check container status
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo "Container is already running. Attaching..."
        docker attach ${CONTAINER_NAME}
    else
        echo "Container exists but is not running. Starting and attaching..."
        docker start -i ${CONTAINER_NAME}
    fi
else
    echo "Creating new container..."
    docker run --name ${CONTAINER_NAME} \
        --gpus all \
        --shm-size=32g \
        -it \
        -v $(pwd):/app \
        -v $SAGEPATH_MOUN \
        -v $HOME/workspace/sage:/data/kera/workspace/sage \
        -e SAGE_ROOT=/sage \
        -e PATH="/sage/local/var/lib/sage/venv-python3.10/bin:/usr/bin:$PATH" \
        -e SAGE_LOCAL=/sage/local \
        -e PYTHONPATH="/sage/local/var/lib/sage/venv-python3.10/lib/python3.10/site-packages:$PYTHONPATH" \
        -e VIRTUAL_ENV="/sage/local/var/lib/sage/venv-python3.10" \
        torch-2.3.0-sage \
        /bin/bash
fi