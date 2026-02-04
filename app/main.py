from fastapi import FastAPI, Depends
from pydantic import BaseModel
from app.predict import TemperaturePredictor
from app.database.session import get_db, init_db
from app.database.models import Prediction
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(
    title="Temperature Prediction API",
    version="1.0",
    lifespan=lifespan
)

predictor = TemperaturePredictor()

class TemperatureInput(BaseModel):
    meantemp: float
    humidity: float
    wind_speed: float
    meanpressure: float

@app.get("/")
def root():
    return {"message": "API Running"}

@app.post("/predict")
def predict_temperature(
    data: TemperatureInput,
    db: Session = Depends(get_db)
):
    input_dict = data.model_dump()
    result = predictor.predict(input_dict)

    prediction = Prediction(
        meantemp=input_dict["meantemp"],
        humidity=input_dict["humidity"],
        wind_speed=input_dict["wind_speed"],
        meanpressure=input_dict["meanpressure"],
        predicted_temp=result
    )

    db.add(prediction)
    db.commit()

    return {"predicted_next_day_temperature": result}

@app.get("/predictions")
def get_predictions(db: Session = Depends(get_db)):
    return db.query(Prediction).all()

@app.delete("/delete")
def delete_all_predictions(db: Session = Depends(get_db)):
    num_deleted = db.query(Prediction).delete()
    db.commit()
    return {"message": f"Successfully deleted {num_deleted} predictions"}

# print('success')