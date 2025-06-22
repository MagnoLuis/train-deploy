from data_model import NLPDataInput, NLPDataOutput
from fastapi import FastAPI, Request
import uvicorn

from s3 import download_dir
import os
import logging
import time

import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

###download and loaing ml model from S3 bucket####
force_download = False  # Set to True to force download
local_path = 'tinybert-disaster-tweet'
model_name = 'tinybert-disaster-tweet'
if not os.path.isdir(local_path) or force_download:
    logging.info(f"Downloading model {model_name} to {local_path}...")
    download_dir(local_path, model_name)
tweeter_model = pipeline('text-classification', model=local_path, device=device)
#########################################    


app = FastAPI()

@app.get("/")
def read_root():
    return {"I am up!!!"}

@app.post("/api/v1/twitter_disaster_classifier")
def twitter_disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = tweeter_model(data.text)
    end  = time.time()
    prediction_time = int(1000*(end - start))
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]
    output = NLPDataOutput(model_name="tinybert-disaster-tweet",
                           text = data.text,
                           labels=labels,
                           scores = scores,
                           prediction_time=prediction_time)
    return output


if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)