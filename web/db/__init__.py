from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class BaseMixin:
    def as_dict(self):
        return {col.name: getattr(self, col.name) for col in self.__table__.columns}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()