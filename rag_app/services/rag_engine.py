import os
import pickle
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

VECTOR_DIR = os.path.join(BASE_DIR, "vector_db")
INDEX_PATH = os.path.join(VECTOR_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(VECTOR_DIR, "chunks.pkl")

CHUNK_SIZE = 500
OVERLAP = 100



# INTERNAL HELPERS

def _load_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def _chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - OVERLAP

    return chunks


def _load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# PUBLIC API
def index_document(pdf_path):
    text = _load_pdf_text(pdf_path)
    chunks = _chunk_text(text)

    model = _load_model()
    embeddings = model.encode(chunks).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def query_rag(question, top_k=3):
    model = _load_model()

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_embedding = model.encode([question]).astype("float32")
    _, indices = index.search(query_embedding, top_k)

    return [chunks[i] for i in indices[0]]


def index_uploaded_pdf(pdf_path):
    index_document(pdf_path)
