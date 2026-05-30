"""
FAISS Vector Store Utilities
Reusable helper functions for building and querying FAISS indexes
"""
import os
import pickle
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv; load_dotenv()

FAISS_INDEX_PATH = "./faiss_index"

def load_documents_from_directory(directory: str, glob: str = "**/*.pdf") -> list:
    """Load all PDF documents from a directory."""
    loader = DirectoryLoader(directory, glob=glob, loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {directory}")
    return documents

def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split documents into chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks: list, save_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Build FAISS index from document chunks."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    return vectorstore

def load_faiss_index(index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Load existing FAISS index from disk."""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"FAISS index loaded from {index_path}")
    return vectorstore

def similarity_search(vectorstore: FAISS, query: str, k: int = 4) -> list:
    """Search for similar documents with scores."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return [{"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in results]

def build_index_from_texts(texts: list, metadatas: list = None) -> FAISS:
    """Build FAISS index from raw text strings (for quick demos)."""
    embeddings = OpenAIEmbeddings()
    if metadatas is None:
        metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
    return FAISS.from_texts(texts, embeddings, metadatas=metadatas)

if __name__ == "__main__":
    sample_texts = [
        "Fire insurance covers damage from accidental fires, lightning strikes, and explosions.",
        "Marine cargo insurance protects goods in transit against loss, damage, or theft.",
        "Workers compensation provides benefits to employees injured during work.",
    ]
    vs = build_index_from_texts(sample_texts)
    results = similarity_search(vs, "What does fire insurance cover?", k=2)
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['content'][:100]}")
