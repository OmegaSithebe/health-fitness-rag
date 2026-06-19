from collections import Counter
import glob
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"

llm = ChatOpenAI(temperature=0.3, model_name=MODEL)  # Slight temperature for helpful responses

KNOWLEDGE_BASE = Path(__file__).parent.parent / "knowledge-base"
TOP_K = 5


def _load_documents() -> list[Document]:
    documents: list[Document] = []
    for folder in glob.glob(str(KNOWLEDGE_BASE / "*")):
        doc_type = os.path.basename(folder)
        for file_path in Path(folder).glob("**/*.md"):
            text = file_path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "filename": file_path.name,
                        "doc_type": doc_type,
                    },
                )
            )
    return documents


def _chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


DOCUMENTS = _chunk_documents(_load_documents())


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_document(query_tokens: Counter[str], document: Document) -> tuple[int, int]:
    doc_tokens = Counter(_tokenize(document.page_content))
    shared_terms = sum(min(query_tokens[token], doc_tokens[token]) for token in query_tokens)
    return shared_terms, len(document.page_content)

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
    query_tokens = Counter(_tokenize(query))
    scored_docs = sorted(
        DOCUMENTS,
        key=lambda document: _score_document(query_tokens, document),
        reverse=True,
    )
    return scored_docs[:TOP_K]


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