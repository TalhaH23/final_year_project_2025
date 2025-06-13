from langchain_openai import ChatOpenAI

light_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=4000)
strong_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4000)

def build_llm(chat_args):
    """Helper function to build an LLM instance with given arguments."""
    return ChatOpenAI()