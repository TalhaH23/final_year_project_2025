from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.chat.chat import build_chat
from app.models import ChatArgs
from web.db.models.pdf import Pdf
from web.db.models.conversation import Conversation
from web.db import get_db

router = APIRouter(prefix="/api/conversations")

@router.post("/{conversation_id}/messages")
async def create_message(conversation_id: str, request: Request, db: Session = Depends(get_db)):
    data = await request.json()
    user_input = data.get("input")

    # Fetch Conversation & Pdf
    conversation = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conversation:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    pdf = db.query(Pdf).filter_by(id=conversation.pdf_id).first()

    chat_args = ChatArgs(
        conversation_id=conversation.id,
        pdf_id=pdf.id,
        streaming=False,
        metadata={
            "conversation_id": conversation.id,
            "pdf_id": pdf.id,
        }
    )

    rag_chain, memory = build_chat(chat_args, db)
    chat_history = memory.chat_memory.messages
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    memory.save_context({"input": user_input}, {"answer": response["answer"]})

    return {"role": "assistant", "content": response["answer"]}
