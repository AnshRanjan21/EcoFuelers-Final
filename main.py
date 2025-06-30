# main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from database import SessionLocal, engine, get_db # create_engine code lives in database.py
import models, schemas
from fastapi.middleware.cors import CORSMiddleware
from led_routers import router as led_router, is_manual_active
from dotenv import load_dotenv
import httpx, os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime, timezone


load_dotenv()

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Adaptive Lighting API", version="1.0")
app.include_router(led_router)

# allow Streamlit (localhost:8501) to call the API during dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WEATHER_API = "https://api.openweathermap.org/data/2.5/weather"
LAT, LON    = "13.08", "80.27"                     #  Chennai

def persist_led_levels(led_ids: list[int], level: int):
    """Stand‑alone DB session per background task."""
    db = SessionLocal()
    try:
        for lid in led_ids:
            db.add(models.BrightnessLevel(led_id=lid, level=level))
        db.commit()
    finally:
        db.close()

def fetch_weather() -> str:
    """
    Return a short weather string, e.g. 'clear sky', 'light rain'.
    Falls back to 'unknown' on error.
    """
    try:
        r = httpx.get(
            WEATHER_API,
            params={"lat": LAT, "lon": LON, "appid": os.getenv("OPENWEATHER_KEY")},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()["weather"][0]["description"]
    except Exception:
        return "unknown"

_groq_model = None
def groq_model():
    global _groq_model
    if _groq_model is None:                 # lazy init, one per process
        _groq_model = init_chat_model("llama3-8b-8192", model_provider="groq")
    return _groq_model

def brightness_via_llm(lux: int, ts: datetime, weather: str) -> int:
    """
    Ask Groq Llama‑3 how bright (0‑100 %) the LED should be.
    Returns an int; falls back to rule‑based 70 % on parse error.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an adaptive‑lighting controller. "
                "Given sensor lux, date, time‑of‑day, and weather, "
                "return only one integer 0‑100 representing the LED brightness %. "
                "Do not output anything else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"lux:{lux}\n"
                f"date:{ts.date()}\n"
                f"time:{ts.strftime('%H:%M')}\n"
                f"weather:{weather}"
            ),
        },
    ]

    try:
        reply = groq_model().invoke(messages).content.strip()
        level = int("".join(filter(str.isdigit, reply)))   # robust parse
        return max(0, min(level, 100))
    except Exception:
        return 70
    

# --- endpoints -----------------------------------------------------------
@app.get("/", status_code=200)
def home():
    return {"message" : "Hello world!"}

@app.post("/readings", response_model=schemas.SensorReadingOut, status_code=201)
def ingest_reading(
    data: schemas.SensorReadingIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    # 1️⃣ store raw reading (unchanged)
    reading = models.SensorReading(**data.model_dump())
    db.add(reading)
    db.commit()
    db.refresh(reading)

    # 2️⃣ which LEDs does this sensor drive?  (unchanged)
    led_rows = (
        db.query(models.Led.id)
          .join(models.sensor_led_map)
          .filter(models.sensor_led_map.c.sensor_id == data.sensor_id)
          .all()
    )
    led_ids = [row.id for row in led_rows]

    # ────────────────────────────────────────────────────────────────
    # ❶  Skip LEDs that are still in *manual* mode
    # ────────────────────────────────────────────────────────────────
    active_led_ids = [lid for lid in led_ids if not is_manual_active(lid)]

    # ❷  If *every* LED tied to this sensor is manual, exit early
    if not active_led_ids:
        return reading          # stop here – no LLM call, no DB write

    # 3️⃣ decide brightness (unchanged logic, but use active_led_ids)
    if not data.people:
        level = 0
    else:
        weather = fetch_weather()
        level   = brightness_via_llm(
            lux=data.lux,
            ts = datetime.now(timezone.utc),
            weather=weather,
        )

    # ❸  Persist brightness only for LEDs that were NOT in manual mode
    background_tasks.add_task(persist_led_levels, active_led_ids, level)
    return reading