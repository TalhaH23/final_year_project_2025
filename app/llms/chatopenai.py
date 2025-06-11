from langchain_openai import ChatOpenAI

# === LLM Setup ===
light_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=4000)
strong_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4000)

def build_llm(chat_args):
    return ChatOpenAI()