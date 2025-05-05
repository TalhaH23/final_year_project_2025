from app.celery import celery_app
import app.celery.tasks.embeddings

if __name__ == "__main__":
    print(celery_app.tasks.keys())