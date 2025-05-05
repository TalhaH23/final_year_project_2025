from web.db import Base, engine
from web.db.models.pdf import Pdf  
from web.db.models.conversation import Conversation  
from web.db.models.message import Message 

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

print("Database tables created.")
