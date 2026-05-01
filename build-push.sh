#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

REPOSITORY_NAME="$1"
IMAGE_TAG="$2"
DOCKER_USERNAME="$3"

# Check if repository name, image tag, and Docker username are provided
if [ -z "$REPOSITORY_NAME" ] || [ -z "$IMAGE_TAG" ] || [ -z "$DOCKER_USERNAME" ]; then
  echo "Usage: $0 <repository_name> <image_tag> <docker_username>"
  exit 1
fi

echo "Building Docker image with the following details:"
echo "Repository Name: $REPOSITORY_NAME"
echo "Image Tag: $IMAGE_TAG"
echo "Docker Username: $DOCKER_USERNAME"

# Full image name including the Docker Hub username
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${REPOSITORY_NAME}:${IMAGE_TAG}"

# Build the Docker image locally using docker 'buildx' for Apple M1/M2 architectures
docker buildx create --use
docker buildx build --platform linux/amd64 -t "${FULL_IMAGE_NAME}" -f Dockerfile --push .

# Check if the build and push was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to build and push the image to ${FULL_IMAGE_NAME}"
  exit 1
fi

echo "Image successfully pushed to ${FULL_IMAGE_NAME}"

# Clean up Docker system
docker system prune -f
