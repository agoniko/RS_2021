# WhoTeach RS

## In this branch we join the results coming form CF, CB and the semantic similarity between texts

* **Objective** This is the backend of the WhoTeach Recommender System.
* **Frameworks:** Sanic, Cloud Run, Docker



## Production endpoint
* https://whoteach-dev-vjywjfifqq-ew.a.run.app/api/rs

## Fire up the func (simple pinging)
* curl -X POST -d '{"type":"ping"}' http://localhost:5001/api/rs
* curl -X POST -d '{"type":"ping"}' https://whoteach-dev-vjywjfifqq-ew.a.run.app/api/rs

## Trigger the training
* curl -X POST -d '{"type":"training"}' http://localhost:5001/api/rs
* curl -X POST -d '{"type":"training"}' https://whoteach-dev-vjywjfifqq-ew.a.run.app/api/rs


## Recommendation by keywords
* curl -X POST -d '{"type":"recommend","userid":2, "keywords":["book","libri"], "usersid":["205"]}' http://localhost:5001/api/rs
* curl -X POST -d '{"type":"recommend","userid":2, "keywords":["book","libri"], "usersid":["205"]}' https://whoteach-dev-vjywjfifqq-ew.a.run.app/api/rs

## Recommendation by keywords based on semantic similarity between texts
* curl -X POST -d '{"type":"cosine", "keywords":["book","libri"], "n":5}' http://localhost:5001/api/rs
* curl -X POST -d '{"type":"cosine", "keywords":["book","libri"], "n":5}' https://whoteach-dev-vjywjfifqq-ew.a.run.app/api/rs


## Package Structure
The package folder is organised as follows:

* **containers**: contains the Dockerfile
* **devops**: contains the scripts:
    * _build.sh_: script for building the docker image
    * _push.sh_: script for pushing the image to Google Container Registry
    * _deploy-cloud-run.sh_: script for deploying the image on Google Cloud Run
* **rs**: contains the package resources, the requirements and the source code
  * **certificates:** contains the ssl certificates for the server (don't work)
  * **resources**: contains the resources for the recommender system
    * _auth_: contains a JSON file with the authentication information for the GCP
    * _encoded_articles_: contains the encoded articles file (this folder can be empty) beacuse the file is downloaded from Google storage at the start of the server)
    * _models-deployed_: contains the deployed models (this folder can be empty) beacuse the file is downloaded from Google storage at the start of the server)
    * _tf_hub_models_: contains the tf hub models, refer to the installation section to have more details
  * _src_: contains the Google storage connection service

## Installation
* Install the requirements.txt file to have the libraries installed
* **IMPORTANT**: the rs/resources/tf_hub_model folder is not uploaded on GitHub, at the very first start of the server this folder is automatically created and the model is downloaded from tensorflow hub and placed here, it's important that this folder is created before executing the deployment scripts so the model will be included in the docker image and the model will be very fast to load.



## Deployment steps
1. Install the [Google SDK](https://cloud.google.com/sdk/docs/install)
2. Install [Docker](https://docs.docker.com/engine/install)

3. in a linux terminal execute the following commands:
  * **gcloud auth login**: this command will open a browser and ask you to authenticate with your Google account
  * **gcloud auth configure-docker**: this command will configure the docker authentication
  * **sudo chmod 666 /var/run/docker.sock**: this command will allow the docker daemon to write to the socket
2. Build the Docker image
    * Run the _build.sh_ script
    * The image will be built and the name will be printed
3. Push the Docker image to Google Container Registry
    * Run the _push.sh_ script
    * The image will be pushed to the registry and the name will be printed
4. Deploy the image on Google Cloud Run
    * Run the _deploy-cloud-run.sh_ script
    * The image will be deployed and the name will be printed