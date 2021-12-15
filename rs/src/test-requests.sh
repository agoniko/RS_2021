#curl -X POST -d '{"type":"ping"}' http://localhost:7071/api/rs
#curl -X POST -d '{"type":"training"}' http://localhost:7071/api/rs
#curl -X POST -d '{"type":"recommend","userid":"2"}' http://localhost:7071/api/rs
#curl -X POST -d '{"type":"recommend","userid":"2", "keywords":["book","libri"]}' http://localhost:5000/api/rs

echo -e "\n1"
curl -k -X POST -d '{"type":"ping"}' https://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs
#echo -e "\n2"
#curl -k -X POST -d '{"type":"training"}' https://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs
echo -e "\n3"
curl -k -X POST -d '{"type":"recommend","userid":"2"}' https://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs
echo -e "\n4"
curl -k -X POST -d '{"type":"recommend", "keywords":["book","libri"]}' https://socialthings-rs-ml.westeurope.cloudapp.azure.com:5000/api/rs
