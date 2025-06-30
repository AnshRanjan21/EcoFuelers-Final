# models.py
from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey, SmallInteger, String, Numeric, Table, BigInteger
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class Sensor(Base):
    __tablename__ = "sensors"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    location  = Column(String(128))

class Led(Base):
    __tablename__ = "leds"
    id       = Column(Integer, primary_key=True, autoincrement=True)
    wattage  = Column(Numeric(5, 2), nullable=False)

# association table
sensor_led_map = Table(
    "sensor_led_map",
    Base.metadata,
    Column("sensor_id", ForeignKey("sensors.id"), primary_key=True),
    Column("led_id",    ForeignKey("leds.id"),    primary_key=True)
)

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id        = Column(BigInteger, primary_key=True, autoincrement=True)
    sensor_id = Column(Integer, ForeignKey("sensors.id"), nullable=False)
    lux       = Column(Integer, nullable=False)
    people    = Column(Boolean, nullable=False)
    ts        = Column(DateTime, default=datetime.utcnow, nullable=False)

class BrightnessLevel(Base):
    __tablename__ = "brightness_levels"
    id     = Column(BigInteger, primary_key=True, autoincrement=True)
    led_id = Column(Integer, ForeignKey("leds.id"), nullable=False)
    level  = Column(SmallInteger, nullable=False)  # 0-100
    ts     = Column(DateTime, default=datetime.utcnow, nullable=False)