# api.py
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
import os
from pinecone import Pinecone
from dotenv import load_dotenv
import tempfile
from scraper import get_all_angelone_support_urls

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
index = pinecone.Index(host=os.getenv("PINECONE_HOST"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_APIKEY"), model="text-embedding-3-small", dimensions=1024)
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

    for chunk in chunks:
        embedding = embeddings.embed_query(chunk.page_content)
        
        record = {
            "id": str(uuid4()),
            "values": embedding,
            "metadata": {
                "page_content": chunk.page_content,
                **chunk.metadata
            }
        }
        index.upsert([record], namespace="support_docs")

    os.remove(tmp_path)
    
    return {
        "message": "Document ingested successfully",
        "chunks": len(chunks)
    }


# Currently only for AngelOne Support URLs
@app.post("/ingest-from-url")
async def ingest_from_url():
    faq_data = get_all_angelone_support_urls()

    for faq_list in faq_data:
        records = []
        for faq in faq_list:
            embedding = embeddings.embed_query(f"{faq['title']} - {faq['content']}")
            
            record = {
                "id": str(uuid4()),
                "values": embedding,
                "metadata":{
                    "title": faq["title"],
                    "source_url": faq["url"],
                    "page_content": faq["content"]
                }
            }
            records.append(record)
        print(len(records))
        index.upsert(records, namespace="support_docs")

    return {
        "message": "URL ingested successfully",
        "records": len(records)
    }