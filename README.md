# Insurance RAG Chatbot 🤖

A Retrieval-Augmented Generation (RAG) chatbot specifically designed for insurance and financial support queries. This chatbot leverages LangChain, OpenAI, and Pinecone to provide accurate, context-aware responses based on insurance documentation and support materials.

## Installation 🚀

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd insurance-rag-chatbot
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_APIKEY=your_openai_api_key_here
   PINECONE_APIKEY=your_pinecone_api_key_here
   PINECONE_HOST=your_pinecone_host_here
   ```

## Usage 📖

### Starting the Chatbot

1. **Run the chatbot**

   ```bash
   chainlit run chatbot.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8000`)

3. **Start chatting** with the insurance support assistant!

### Ingesting Documents

#### From PDF Files

```bash
# Start the ingestion service
uvicorn ingest:app --reload

# Upload a PDF file via the API endpoint
curl -X POST "http://localhost:8000/ingest-from-doc" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
```

#### From AngelOne Support Website (example here)

```bash
# Ingest support content from AngelOne (currently supported)
curl -X POST "http://localhost:8000/ingest-from-url"
```
