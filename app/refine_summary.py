from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=2000)

def refine_summary(docs):
    """Generate a refined summary from a list of Document chunks."""

    initial_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Write an initial summary of the following content:

        {text}

        SUMMARY:
        """
    )

    refine_prompt = PromptTemplate(
        input_variables=["existing_summary", "text"],
        template="""
        You have an existing summary and a new section of the document.

        Existing summary:
        {existing_summary}

        New content:
        {text}

        Refine the summary based on the new content:
        """
    )

    # Start with initial chunk
    initial_input = {"text": docs[0].page_content}
    summary = llm.invoke(initial_prompt.format_prompt(**initial_input).to_string()).content

    # Refine with each subsequent chunk
    for doc in docs[1:]:
        refine_input = {
            "existing_summary": summary,
            "text": doc.page_content
        }
        summary = llm.invoke(refine_prompt.format_prompt(**refine_input).to_string()).content

    return summary