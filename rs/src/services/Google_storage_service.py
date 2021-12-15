from google.cloud import storage
import tensorflow_hub as hub
import tensorflow_text
import logging
import tensorflow as tf
from gcsfs import GCSFileSystem
import os
from os import listdir
from os.path import isfile, join



class Gstorage:
    def __init__(self):
       #load the bucket client
        self.bucket_name = "wooteach-rs"
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)
        self.local_path = "resources/models-deployed/"
        self.storage_path = "models/"

    def upload_model_and_csv(self):
        files = [f for f in listdir(self.local_path) if isfile(join(self.local_path, f))]
        for file in files:
            self.upload_file(self.storage_path + file, self.local_path + file)

    def upload_file(self, storage_path, local_path):
        blob = self.bucket.blob(storage_path)
        blob.upload_from_filename(local_path)

    def download_files(self):
        files = ['model-ratings.h5','resource-dict-encoded.csv','users-dict-encoded.csv']
        for file in files:
            self.download_file(self.storage_path + file, self.local_path + file)


    def download_file(self, storage_path, local_path):
        blob = self.bucket.blob(storage_path)
        blob.download_to_filename(local_path)


