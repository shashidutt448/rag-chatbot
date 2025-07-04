import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_document(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        print("Page text:", page_text)  # 🔍 Debug
        text += page_text or ""
    return text


def clean_text(text):
    text = re.sub(r"\\n+", "\\n", text)
    text = re.sub(r"\\s{2,}", " ", text)
    return text.strip()

def split_text(text):
    print("Text Length:", len(text))  # Debug
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)
    print("Chunks created:", len(chunks))  # Debug
    return chunks

