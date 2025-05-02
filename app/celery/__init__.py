# app/celery/__init__.py
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()
REDIS_URI = os.getenv("REDIS_URI", "redis://127.0.0.1:6379/0")

celery_app = Celery(
    "worker",
    broker=REDIS_URI,
    backend=REDIS_URI,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)
