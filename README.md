# RS_WT

## In this branch we join the two results coming form CF and CB

## Run
* func start

## Production endpoint
* http://socialthings-rs.azurewebsites.net/api/rs

## Fire up the func (simple pinging)
* curl -X POST -d '{"type":"ping"}' http://localhost:7071/api/rs
* curl -X POST -d '{"type":"ping"}' http://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs

## Trigger the training
* curl -X POST -d '{"type":"training"}' http://localhost:7071/api/rs
* curl -X POST -d '{"type":"training"}' http://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs

## Request recommendation
* curl -X POST -d '{"type":"recommend","userid":"2"}' http://localhost:7071/api/rs
* curl -X POST -d '{"type":"recommend","userid":"2"}' http://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs

## Recommendation by keywords
* curl -X POST -d '{"type":"recommend","userid":"2", "keywords":["book","libri"]}' http://localhost:5000/api/rs
* curl -X POST -d '{"type":"recommend","userid":"2", "keywords":["book","libri"]}' http://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs

## Create commands (https://docs.microsoft.com/en-us/azure/azure-functions/functions-create-first-azure-function-azure-cli)
* az login
* az group create \
	--name AzureFunctions-RS \
	--location westeurope
* az storage account create \
	--name socialthingsrs \
	--location westeurope \
	--resource-group AzureFunctions-RS \
	--sku Standard_LRS
* az functionapp create \
	--resource-group AzureFunctions-RS \
	--os-type Linux \
	--consumption-plan-location westeurope \
	--runtime python \
	--runtime-version 3.7 \
	--functions-version 2 \
	--name SocialThings-RS \
	--storage-account socialthingsrs

## Production deploy
* func azure functionapp publish SocialThings-RS

## MYSQL
* Enable port 3306 on Windows firewall

## Cronjob for automatic training at 4AM Europe/Rome (actually machine time is UTC)
0 2 * * * curl -X POST -d '{"type":"training"}' http://localhost:5000/api/rs

