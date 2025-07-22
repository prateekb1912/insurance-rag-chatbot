from typing import TypedDict
import chainlit as cl
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

class RAGState(TypedDict):
    question: str
    context: str
    response: str

pinecone = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
index = pinecone.Index(host=os.getenv("PINECONE_HOST"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_APIKEY"), model="text-embedding-3-small", dimensions=1024)

def retrieve_documents(query, k=4):
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )

    print(results)
    
    documents = []
    for match in results.matches:
        documents.append({
            'page_content': match.metadata.get('text', ''),
            'metadata': match.metadata
        })
    
    return documents

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_APIKEY"))
prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful support assistant which has expertise in understanding customer support documentation and answering questions accordingly.
        The user will ask you some financial questions. You have to use the context provided to answer the question.
        If the context is not helpful or you are unsure about the answer, you must respond with "I don't know".

        Context: {context}

        Question: {question}
    """
)

def retrieve_node(state):
    docs = retrieve_documents(state["question"], k=4)
    context = "\n\n".join([doc['page_content'] for doc in docs])
    return {
        "question": state["question"], 
        "context": context
    }

def generate_node(state):
    formatted = prompt.format(**state)
    result = llm.invoke(formatted)
    return {
        "response": result.content
    }


def build_rag_agent():
    builder = StateGraph(state_schema=RAGState)
    
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    builder.set_entry_point("retrieve")

    rag_agent = builder.compile()

    return rag_agent

@cl.on_message
async def handler(message: cl.Message):
    rag_agent = build_rag_agent()
    state = await rag_agent.ainvoke({"question": message.content})
    await cl.Message(content=state["response"]).send()
