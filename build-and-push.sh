#!/usr/bin/env bash

# Read image name and version from YAML file
DOCKERFILE=$1
IMAGE=$(yq r image.yaml image)
VERSION=$(yq r image.yaml version)
REGISTRY=$(yq r image.yaml registry)
IMAGE_TAG="${IMAGE}:${VERSION}"

# Build Docker image
docker build --platform=linux/amd64 -t "${IMAGE_TAG}" -f "${DOCKERFILE}" .

docker tag "${IMAGE_TAG}" "jfrog.fkinternal.com/n200m-common-infra/${IMAGE_TAG}"

# Push Docker image to registry
docker push "${REGISTRY}/${IMAGE_TAG}"
