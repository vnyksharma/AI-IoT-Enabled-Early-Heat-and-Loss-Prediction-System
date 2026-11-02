
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

# Database will be created inside the backend folder.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "motorai_new.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS motor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                temperature_c REAL NOT NULL,
                accel_x_g REAL,
                accel_y_g REAL,
                accel_z_g REAL,
                vibration_rms_g REAL
            )
        """)


def save_reading(data):
    received_at = data.get("received_at")

    if not received_at:
        received_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO motor_readings (
                received_at,
                temperature_c,
                accel_x_g,
                accel_y_g,
                accel_z_g,
                vibration_rms_g
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            received_at,
            data["temperature_c"],
            data.get("accel_x_g"),
            data.get("accel_y_g"),
            data.get("accel_z_g"),
            data.get("vibration_rms_g")
        ))


def get_latest_reading():
    with get_connection() as connection:
        row = connection.execute("""
            SELECT *
            FROM motor_readings
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        return dict(row) if row else None


def get_recent_readings(limit=100):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM motor_readings
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]


initialize_database()
