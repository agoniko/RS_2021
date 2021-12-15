#!/bin/bash
# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Setting project id
PROJECT_ID="whoteach-dev"
gcloud config set project ${PROJECT_ID}

# Enabling required services
gcloud services enable "containerregistry.googleapis.com"

# Defining constants
REPOSITORY="whoteach-dev"
REMOTE_IMAGE=${REPOSITORY}

HOSTNAME=eu.gcr.io

# Pushing image to Google Container Registry
docker tag ${REPOSITORY} ${HOSTNAME}/${PROJECT_ID}/${REMOTE_IMAGE}
docker push ${HOSTNAME}/${PROJECT_ID}/${REMOTE_IMAGE}