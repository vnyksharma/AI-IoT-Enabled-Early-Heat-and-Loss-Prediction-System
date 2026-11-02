
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.database import (
    save_reading,
    get_latest_reading,
    get_recent_readings
)

app = FastAPI(
    title="MotorAI API",
    description="Motor temperature and vibration monitoring system",
    version="1.0.0"
)


class MotorReading(BaseModel):
    temperature_c: float = Field(
        ge=-55,
        le=125
    )

    accel_x_g: float | None = None
    accel_y_g: float | None = None
    accel_z_g: float | None = None
    vibration_rms_g: float | None = None


@app.get("/")
def home():
    return {
        "message": "MotorAI API is running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/telemetry")
def receive_telemetry(reading: MotorReading):
    data = reading.model_dump()

    data["received_at"] = datetime.now(timezone.utc).isoformat()

    try:
        save_reading(data)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Unable to save sensor reading"
        ) from error

    return {
        "message": "Sensor data received",
        "data": data
    }


@app.get("/latest")
def latest():
    reading = get_latest_reading()

    if reading is None:
        return {"message": "No sensor readings available"}

    return reading


@app.get("/history")
def history(limit: int = 100):
    limit = max(1, min(limit, 10000))
    readings = get_recent_readings(limit)

    return {
        "count": len(readings),
        "readings": readings
    }
