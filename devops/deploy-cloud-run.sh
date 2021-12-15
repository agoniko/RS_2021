#!/bin/bash
# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx

# Setting project id
PROJECT_ID="whoteach-dev"
gcloud config set project ${PROJECT_ID}

# Enabling required services
gcloud services enable "run.googleapis.com"

# Constants
HOSTNAME="eu.gcr.io"
REPOSITORY="whoteach-dev"
REMOTE_IMAGE=${HOSTNAME}/${PROJECT_ID}/${REPOSITORY}

COMPONENT=${REPOSITORY}
SERVICE_ACCOUNT="wooteach-rs@${PROJECT_ID}.iam.gserviceaccount.com"

REGION=europe-west1

MEMORY=4Gi
CPU=2
PORT=5001
MIN_INSTANCES=2

# Deploy
gcloud run deploy \
    ${COMPONENT} \
    --project ${PROJECT_ID} \
    --platform managed \
    --region  ${REGION}\
    --image ${REMOTE_IMAGE} \
    --service-account ${SERVICE_ACCOUNT} \
    --memory ${MEMORY} \
    --cpu=${CPU} \
    --port ${PORT} \
    --concurrency=200 \
    --timeout 10m \
    --min-instances ${MIN_INSTANCES} \
    --allow-unauthenticated