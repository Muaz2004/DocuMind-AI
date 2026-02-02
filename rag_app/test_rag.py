from services.rag_engine import index_document, query_rag

PDF_PATH = "test_docs/sample.pdf"

index_document(PDF_PATH)

results = query_rag("What does the company do?")
for r in results:
    print(r)
