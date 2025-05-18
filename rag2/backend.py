#imports
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# FIle Paths and Model Names
PDF_PATH = r"C:\Users\sakth\OneDrive\Desktop\rag2\1900-2025.pdf"  
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"  


app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=10
    )
    return HuggingFacePipeline(pipeline=pipe)

def load_chunks():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.from_documents(chunks, embeddings)

def get_rag_chain():
    chunks = load_chunks()
    vectordb = get_vector_store(chunks)
    retriever = vectordb.as_retriever()
    llm = load_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return chain

rag_chain = get_rag_chain()

@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer = rag_chain.run(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
