import PyPDF2
import os

# Path to your docs folder
docs_path = "docs/"

# List PDF files in the folder
pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]

# For now, just use the first PDF
pdf_file = pdf_files[0]

# Open and read PDF
pdf_text = ""
with open(os.path.join(docs_path, pdf_file), "rb") as file:
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"

print("PDF loaded successfully!")
print(f"First 500 characters:\n{pdf_text[:500]}")


# -------- CHUNKING STEP --------

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  

    return chunks


chunks = chunk_text(pdf_text)

print(f"\nTotal chunks created: {len(chunks)}")
print("\nSample chunk:\n")
print(chunks[0])



# -------- EMBEDDINGS STEP --------

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


# Create embeddings for all chunks
print("\nCreating embeddings... (this may take a few seconds)")
chunk_embeddings = get_embeddings(chunks)

print("Embeddings created successfully!")
print(f"Number of embeddings: {len(chunk_embeddings)}")
print(f"Length of one embedding vector: {len(chunk_embeddings[0])}")


# -------- RETRIEVAL STEP --------

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_relevant_chunks(question, chunks, chunk_embeddings, top_k=3):
    # Embed the question
    question_embedding = get_embeddings([question])[0]

    # Compare with all chunks
    similarities = [
        cosine_similarity(question_embedding, emb)
        for emb in chunk_embeddings
    ]

    # Get top matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [chunks[i] for i in top_indices]


# Test question
question = "What does the company AlphaTech AI do?"

relevant_chunks = retrieve_relevant_chunks(question, chunks, chunk_embeddings)

print("\nTop relevant chunks:\n")
for i, chunk in enumerate(relevant_chunks, 1):
    print(f"--- Result {i} ---")
    print(chunk)
    print()
