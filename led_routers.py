# api/led_routes.py  – actual endpoints
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import schemas as s
import models                 
from database import get_db
from threading import Lock

# ── manual‑override registry ───────────────────────────────
manual_led: dict[int, datetime] = {}   # <LED id, timestamp‑UTC>
_manual_lock = Lock()  

router = APIRouter(prefix="/api", tags=["leds"])

def is_manual_active(led_id: int, *, expiry_hours: int = 2) -> bool:
    now_utc = datetime.now(timezone.utc)
    expiry  = timedelta(hours=expiry_hours)

    with _manual_lock:
        ts = manual_led.get(led_id)
        if ts is None:
            return False

        # Ensure ts is timezone-aware (UTC); fallback if it isn't
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        if now_utc - ts >= expiry:
            manual_led.pop(led_id, None)
            return False

        return True

def clear_manual_override(led_id: int) -> None:
    with _manual_lock:
        manual_led.pop(led_id, None)          # no KeyError if missing

@router.get("/leds", response_model=list[s.LedOut])
def list_leds(db: Session = Depends(get_db)):
    """
    Return every LED fixture (id + wattage) for the dropdown menu.
    """
    return db.query(models.Led).order_by(models.Led.id).all()

@router.get("/leds/{led_id}/history", response_model=list[s.BrightnessPoint])
def led_history(
    led_id: int,
    hours: int = Query(24, ge=1, le=720),   # user-tunable, but safe-bounded
    db: Session = Depends(get_db),
):
    """
    Brightness timeline for the past ⧖ *hours*.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    rows = (
        db.query(models.BrightnessLevel)
          .filter(
              models.BrightnessLevel.led_id == led_id,
              models.BrightnessLevel.ts >= cutoff,
          )
          .order_by(models.BrightnessLevel.ts)
          .all()
    )

    # extra guard: 404 if LED id doesn’t exist at all
    if not rows and not db.query(models.Led.id).filter_by(id=led_id).first():
        raise HTTPException(status_code=404, detail="LED not found")

    return rows

@router.post("/override_brightness", response_model=s.BrightnessPoint, status_code=201,)
def override_brightness(
    payload: s.BrightnessOverrideIn,
    db: Session = Depends(get_db),
):
    
    if not db.query(models.Led.id).filter_by(id=payload.led_id).first():
        raise HTTPException(status_code=404, detail="LED not found")

    row = models.BrightnessLevel(
        led_id=payload.led_id,
        level=payload.level,
        ts = datetime.now(timezone.utc)           
    )
    db.add(row)
    db.commit()
    db.refresh(row)                     
    return row

@router.get("/leds/{led_id}/sensor_history", response_model=list[s.SensorPoint])
def led_sensor_history(
    led_id: int,
    hours: int = Query(168, ge=1, le=720),     # default 7 d, max 30 d
    db: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    sensor_ids = (
        db.query(models.sensor_led_map.c.sensor_id)
          .filter(models.sensor_led_map.c.led_id == led_id)
          .all()
    )
    if not sensor_ids:
        raise HTTPException(status_code=404, detail="LED not mapped to any sensor")

    rows = (
        db.query(models.SensorReading)
          .filter(
              models.SensorReading.sensor_id.in_([sid for sid, in sensor_ids]),
              models.SensorReading.ts >= cutoff,
          )
          .order_by(models.SensorReading.ts)
          .all()
    )
    return rows

@router.post(
    "/register_manual",                     # ← NEW endpoint
    response_model=s.ManualRegisterOut,
    status_code=201,
)
def register_manual(payload: s.ManualRegisterIn):
    """
    Remember that this LED is now in manual mode.
    """
    with _manual_lock:                      # atomic write
        manual_led[payload.led_id] = payload.ts_utc

    return payload

@router.post(
    "/auto_brightness",
    status_code=200,
    summary="Return LED to auto‑brightness mode",
)
def auto_brightness(payload: s.AutoModeIn):
    """
    Remove this LED from the manual‑override registry.
    Idempotent: returns 200 even if the LED wasn’t in manual mode.
    """
    clear_manual_override(payload.led_id)
    return {"led_id": payload.led_id, "mode": "auto"}