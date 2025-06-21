from data_model import NLPDataInput, NLPDataOutput
from fastapi import FastAPI
from fastapi import Request
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"I am up!!!"}

@app.post("/api/v1/twitter_disaster_classifier")
def twitter_disaster_classifier(data: NLPDataInput):
    return data


print("API is running...")
if __name__ == "__main__":
    uvicorn.run(app="app:app", port=8000, reload=True)