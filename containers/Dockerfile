##########################################################
# Base image #############################################
##########################################################
FROM python:3.8.10 as rs-2021-builder

# Upgrading pip and installing requirements ##############
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

##########################################################
# Prod image #############################################
##########################################################
FROM python:3.8.10-slim as rs-2021-prod

# Setting container working directory ####################
WORKDIR /app

# Copying from previous stage ############################
COPY --from=rs-2021-builder /usr/local/ /usr/local/

#  Copying resources and source ##########################
COPY ./resources/ ./resources/
COPY ./src/ ./src/
COPY ./main.py .

# Exposing port ##########################################
EXPOSE 5001

# Docker image label depends on commit ###################
ARG GIT_COMMIT=unspecified
LABEL git_commit=$GIT_COMMIT

CMD ["python3", "./main.py"]