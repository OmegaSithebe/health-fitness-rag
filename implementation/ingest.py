import os
import glob
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Use a separate index for fitness-health
index_name = "fitness-health-main"

# Create if missing
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index
index = pc.Index(index_name)

NAMESPACE = "fitness-health"


def fetch_documents():
    """Load all markdown files from the knowledge base."""
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)


def embed_and_prepare(chunks):
    """Generate embeddings and prepare vectors for Pinecone."""
    vectors = []
    for i, doc in enumerate(chunks):
        vec = embeddings.embed_query(doc.page_content)
        vectors.append({
            "id": f"doc-{i}",
            "values": vec,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "text": doc.page_content,
                "doc_type": doc.metadata.get("doc_type", ""),
                "filename": os.path.basename(doc.metadata.get("source", ""))
            }
        })
    return vectors


def insert_parallel(vectors, batch_size=100):
    """Insert vectors into Pinecone in parallel."""
    async def run_upserts():
        with ThreadPoolExecutor(max_workers=5) as executor:
            loop = asyncio.get_running_loop()
            tasks = []
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                tasks.append(loop.run_in_executor(executor, index.upsert, batch, NAMESPACE))
            await asyncio.gather(*tasks)

    asyncio.run(run_upserts())


if __name__ == "__main__":
    print("📚 Loading documents from knowledge base...")
    docs = fetch_documents()
    print(f"✅ Loaded {len(docs)} documents")

    print("✂️ Creating chunks...")
    chunks = create_chunks(docs)
    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Generating embeddings...")
    vectors = embed_and_prepare(chunks)
    print(f"✅ Generated {len(vectors)} vectors")

    print("📤 Inserting into Pinecone...")
    insert_parallel(vectors)
    print(f"🚀 Ingestion complete: {len(vectors)} vectors inserted (namespace={NAMESPACE})")