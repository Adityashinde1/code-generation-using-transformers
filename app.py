from fastapi import FastAPI, File
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from code_generation.pipeline.train_pipeline import TrainPipeline
from code_generation.pipeline.prediction_pipeline import ModelPredictor
from code_generation.constants import *

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text: str):
    try:
        prediction_pipeline = ModelPredictor()

        code = prediction_pipeline.run_pipeline(src=text)

        return code

    except Exception as e:
        return Response(f"Error Occurred! {e}")

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)