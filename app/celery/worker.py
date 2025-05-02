# app/celery/worker.py
from app.celery import celery_app

# Import all task modules here so Celery registers them
import app.celery.tasks.embeddings

# Optionally verify registration
if __name__ == "__main__":
    print(celery_app.tasks.keys())