from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import load_model, make_prediction, create_prediction_dataset

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    load_model()
    yield

app = FastAPI(
    title="Weather API",
    version="0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def welcome():
    return {"msg": "hello"}

@app.get("/predict")
def predict_weather():
    model = load_model()
    dataset = create_prediction_dataset()

    with make_prediction(model=model, input_data_tensor=dataset) as data:
        pass

    return {
        "prediction": [int(v) for v in data[-1]],
        "length":len(data[-1]),
        # "last":int(data[-1])
    }