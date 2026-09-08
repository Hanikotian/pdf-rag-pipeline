# PDF RAG Pipeline

A production-oriented Retrieval-Augmented Generation (RAG) system for asking questions about PDF documents.

The system processes a PDF, creates page-aware chunks, generates embeddings, stores them in a FAISS vector index, retrieves relevant content, reranks the retrieved chunks using a Cross-Encoder, and generates a grounded answer using Google Gemini with source page citations.

## Architecture

```text
PDF Document
     |
     v
PDF Ingestion
     |
     v
Document Metadata
     |
     v
Page-Aware Chunking
     |
     v
Embeddings
     |
     v
FAISS Vector Store
     |
     v
Semantic Retrieval
     |
     v
Cross-Encoder Reranking
     |
     v
Relevant Context
     |
     v
Google Gemini
     |
     v
Grounded Answer
     |
     v
Source / Page Citations
```

## Features

- Generic PDF ingestion
- Page-level text extraction using PyMuPDF
- Page-aware document chunking
- Local semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- Cross-Encoder reranking
- Gemini-powered grounded answer generation
- Structured LLM responses
- Source and page citation mapping
- Persistent FAISS indexes
- Pydantic data models
- Automated unit tests
- Integration tests for LLM functionality
- Streamlit application layer

## Tech Stack

- Python
- PyMuPDF
- Sentence Transformers
- FAISS
- Cross-Encoder
- Google Gemini
- Pydantic
- Streamlit
- Pytest
