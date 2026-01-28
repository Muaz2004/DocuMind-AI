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
