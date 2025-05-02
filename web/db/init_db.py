from web.db import Base, engine
from web.db.models.pdf import Pdf  # import all your models here

# Create all tables
Base.metadata.create_all(bind=engine)

print("✅ Database tables created.")
