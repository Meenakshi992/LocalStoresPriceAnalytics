import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import json

def extract_text_from_file(uploaded_file) -> str:
    """Extracts plain text from an uploaded PDF or TXT file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text("text") for page in doc)
    return uploaded_file.read().decode("utf-8", errors="ignore")


def step2_clean_text(text: str, remove_urls: bool, normalize_space: bool, lowercase: bool) -> str:
    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", "", text)
    if normalize_space:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
    if lowercase:
        text = text.lower()
    return text.strip()

def ingest_document_chunks(
    collection,
    uploaded_file,
    store_tag: str,
    chunk_size: int = 600,
    chunk_overlap: int = 100
) -> dict:
    """Chunks uploaded document text and embeds it into ChromaDB."""
    raw_text = extract_text_from_file(uploaded_file)
    if not raw_text.strip():
        return {"chunks_added": 0, "total_chars": 0}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(raw_text)
    
 # Extract the top header (first 250 chars) so every chunk preserves the flyer dates
    header_snippet = raw_text.strip()[:250].replace("\n", " ")

    docs, metadatas, ids = [], [], []
    for idx, chunk in enumerate(chunks):
        docs.append(f"[Flyer from {store_tag} | Header: {header_snippet}] {chunk}")
        metadatas.append(
            {
                "source": uploaded_file.name,
                "store": store_tag,
                "category": "Uploaded Flyer",
                "distance_km": 0.0,
                "price": 0.0,
                "chunk_index": idx,
                "chunk_size": chunk_size,
            }
        )
        ids.append(f"doc_{uploaded_file.name}_c{idx}")

    collection.upsert(documents=docs, metadatas=metadatas, ids=ids)
    return {"chunks_added": len(chunks), "total_chars": len(raw_text)}