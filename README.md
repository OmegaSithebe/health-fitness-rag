# 💪 Fitness & Health RAG Assistant

An intelligent Retrieval-Augmented Generation (RAG) chatbot designed to provide personalized fitness, nutrition, sleep, and health guidance.

## 🌟 Features

- **Comprehensive Knowledge Base**: Covers workouts, nutrition, sleep, stress management, and general health
- **Context-Aware Responses**: Uses RAG to provide accurate, evidence-based information
- **Conversational Interface**: Natural conversation flow with history awareness
- **Evaluation Framework**: Built-in retrieval and answer quality evaluation
- **Deployment Ready**: Configured for Hugging Face Spaces deployment

## 🏗️ Architecture

The system follows a RAG (Retrieval-Augmented Generation) architecture:

1. **User Query**: User submits a question through the Gradio interface
2. **Query Processing**: Query is resolved with conversation history context
3. **Vector Retrieval**: Semantic search retrieves relevant chunks from Pinecone vector database
4. **Context Assembly**: Retrieved chunks are combined and formatted for the LLM
5. **Response Generation**: OpenAI GPT-4 generates context-aware, evidence-based responses
6. **Context Display**: Retrieved sources are shown to the user for transparency

### Architecture Diagram
