#!/bin/bash
# -e: Exit immediately if a command exits with a non-zero status, -v: Print shell input lines as they are read.
set -evx
# Locations
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MAIN_DIR="${SCRIPT_DIR}/../"

# Locations and source image
DOCKERFILE_LOCATION="${MAIN_DIR}/containers/Dockerfile"
CONTEXT="${MAIN_DIR}/rs"

REPOSITORY="whoteach-dev"

# Building docker image
docker build  --file "${DOCKERFILE_LOCATION}" \
              --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
              --tag "${REPOSITORY}" \
              "${CONTEXT}"