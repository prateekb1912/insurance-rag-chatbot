import chainlit as cl
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pinecone import Pinecone
from typing import TypedDict

load_dotenv()

class ChatState(TypedDict):
    user_input: str
    question: str
    context: str
    response: str
    chat_history: list[str]

pinecone = Pinecone(api_key=os.getenv("PINECONE_APIKEY"))
index = pinecone.Index(host=os.getenv("PINECONE_HOST"))
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_APIKEY"), 
    model="text-embedding-3-small", 
    dimensions=1024,
)

def retrieve_documents(query, k=4):
    query_embedding = embeddings.embed_query(query)
    
    results = index.query(
        vector=query_embedding,
        top_k=4,
        namespace="support_docs",
        include_metadata=True
    )
    
    documents = []
    for match in results.matches:
        documents.append({
            'page_content': match.metadata.get('page_content', ''),
            'title': match.metadata.get('title', ''),
            'source_url': match.metadata.get('source_url', '')
        })
    
    return documents

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_APIKEY"))

rephrase_prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful support assistant which has expertise in understanding customer support documentation and answering questions accordingly.
        The user will ask you some financial questions. You have to use the context provided to answer the question.
        You are also provided the chat history to help with any follow-ups or context.
        Your task is to rephrase the user's question in a way that is more clear and concise and able to be searched efficiently.
        Make sure the title summarizes the question efficiently and concisely and is always a question.
        You also have access to the chat history to help with any follow-ups or context. Use that context to rephrase the question.

        Example:
        User Input: i'm unable to add funds to my account. please help
        Rephrase: Why am I unable to add funds?

        User Input: i placed a withdraw request but it's not showing up in the dashboard. please help
        Rephrase: When will I get money in my bank account after placing a fund withdrawal request?

        Chat History: {chat_history}
        User Input: {user_input}

        Only return the rephrased question, no other text.
    """
)

prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful support assistant which has expertise in understanding customer support documentation and answering questions accordingly.
        The user will ask you some financial questions. You have to use the context provided to answer the question.
        If the context is not helpful or you are unsure about the answer, you must respond with "I don't know".
        If the message is not a question and just a basic salutation or a follow-up on the previous response, respond appropriately.
        If the source_url is provided, you must include it in the response.

        Context: {context}

        Question: {question}
    """
)


def rephrase_node(state: ChatState) -> ChatState:
    formatted = rephrase_prompt.format(**state)
    result = llm.invoke(formatted)
    full_query = result.content.strip()

    return {
        "question": full_query,
        "chat_history": state["chat_history"]
    }

def retrieve_node(state: ChatState) -> ChatState:
    docs = retrieve_documents(state["question"], k=4)
    context = "\n\n".join([f"Title: {doc['title']}\nSource: {doc['source_url']}\nContent: {doc['page_content']}" for doc in docs])
    return {
        "question": state["question"], 
        "context": context
    }

def generate_node(state: ChatState) -> ChatState:
    formatted = prompt.format(**state)
    result = llm.invoke(formatted)

    updated_history = state.get("chat_history", []) + [f"User: {state['question']}", f"Assistant: {result.content}"]

    return {**state, "response": result.content, "chat_history": updated_history}


def build_rag_agent():
    builder = StateGraph(state_schema=ChatState)
    
    builder.add_node("rephrase", rephrase_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge("rephrase", "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    builder.set_entry_point("rephrase")

    rag_agent = builder.compile()

    return rag_agent

@cl.on_message
async def handler(message: cl.Message):
    rag_agent = build_rag_agent()
    result = await rag_agent.ainvoke({"user_input": message.content, "chat_history": []})
    await cl.Message(content=result["response"]).send()
