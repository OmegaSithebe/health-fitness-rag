import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Connect to Pinecone index
index_name = "fitness-health-main"
NAMESPACE = "fitness-health"

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    namespace=NAMESPACE
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(temperature=0.3, model_name=MODEL)  # Slight temperature for helpful responses

SYSTEM_PROMPT = """
You are a knowledgeable, supportive, and friendly fitness and health assistant. Your goal is to help people improve their health and fitness journey.

Key principles:
- Be encouraging and motivational without being unrealistic
- Provide evidence-based information
- Always emphasize safety and listening to your body
- Recommend consulting healthcare professionals when appropriate
- Give practical, actionable advice
- Tailor advice to different fitness levels

Context:
{context}

When providing advice, always:
1. Acknowledge the person's question
2. Provide clear, actionable steps
3. Include safety considerations
4. Offer modifications for different levels
5. Be supportive and encouraging
"""


def resolve_query(query: str, history: list):
    """Resolve query with conversation history."""
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query


def fetch_context(query: str) -> list[Document]:
    """Retrieve relevant documents from the vector store."""
    return retriever.invoke(query)


def answer_question(query: str, history=None):
    """Generate answer using retrieved context and conversation history."""
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    return response.content, docs