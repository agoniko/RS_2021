sudo certbot --nginx -d socialthings-rs-ml.westeurope.cloudapp.azure.com
cd /etc/letsencrypt/live/socialthings-rs-ml.westeurope.cloudapp.azure.com
cp cert.pem privkey.pem /home/rs-whoteach/RS_gat/rs/certificates