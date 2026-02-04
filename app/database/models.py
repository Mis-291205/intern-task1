from sqlalchemy import Column, Integer, Float, DateTime
# from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    meantemp = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    meanpressure = Column(Float)

    predicted_temp = Column(Float)

    created_at = Column(DateTime, default=lambda: datetime.now(ZoneInfo("Asia/Jakarta")))
