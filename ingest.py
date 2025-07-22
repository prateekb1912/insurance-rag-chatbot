# api.py
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Request
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import tempfile

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
index = pinecone.Index(host=os.getenv("PINECONE_HOST"))
app = FastAPI()

@app.post("/ingest-from-doc")
async def ingest_from_doc(file: UploadFile = File(...)):
    tmp_path = tempfile.mktemp(suffix=".pdf")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    records = [
        {
            "id": str(uuid4()),
            "chunk_text": text.page_content
        }
            for text in chunks
    ]

    index.upsert_records("__default__", records)
    
    return {
        "message": "Document ingested successfully",
        "chunks": len(chunks)
    }


@app.post("/ingest-from-url")
async def ingest_from_url(request: Request):
    url = request.url
    loader = UnstructuredURLLoader(urls=[url], headers={"User-Agent": "Mozilla/5.0"})
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    
    records = [
        {
            "id": str(uuid4()),
            "chunk_text": text.page_content
        }
        for text in chunks
    ]

    index.upsert_records("__default__", records)
    
    return {
        "message": "URL ingested successfully",
        "chunks": len(chunks),
        "url": url
    }