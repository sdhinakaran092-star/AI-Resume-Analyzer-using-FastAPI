'''import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# Correct imports for latest llama-index
from llama_index.core import VectorStoreIndex
#from llama_index import SimpleDirectoryReader
from llama_index.core import SimpleDirectoryReader

#from llama_index import VectorStoreIndex
#from llama_index.indices.vector_store.base import VectorStoreIndex
#from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index import ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.prompts import PromptTemplate

 


# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")  # Gemini API key

# Initialize Gemini LLM
llm = Gemini(model="models/gemini-1.5-flash", api_key=gemini_api_key)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

# Load and index the resume file (adjust path & filename)
documents = SimpleDirectoryReader(input_files=["./data/resume.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# FastAPI app
app = FastAPI()

# Pydantic model for request
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_resume(request: QueryRequest):
    # Structured prompt to ensure answers are strictly from resume
    structured_prompt = f"""
You are an assistant that answers questions strictly based on the user's resume.
Only respond using the information available in the resume.
If the information is not available, say "The information is not available in the resume."

Question: {request.query}
Answer:"""
    
    response = query_engine.query(structured_prompt)
    return {
        "query": request.query,
        "response": str(response).strip()
    }'''


'''import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Configure LLM and embedding model globally
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=gemini_api_key)
Settings.embed_model = "local"

# Load documents and create index
documents = SimpleDirectoryReader(input_files=["./data/resume.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_resume(request: QueryRequest):
    prompt = f"""
You are an assistant that answers questions strictly based on the user's resume.
Only respond using the information available in the resume.
If the information is not available, say "The information is not available in the resume."

Question: {request.query}
Answer:"""

    response = query_engine.query(prompt)
    return {
        "query": request.query,
        "response": str(response).strip()
    }
'''


import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Configure LLM and embedding model globally
Settings.llm = Gemini(model="models/gemini-1.5-flash", api_key=gemini_api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load documents and create index
documents = SimpleDirectoryReader(input_files=["./data/resume.txt"]).load_data()
index = VectorStoreIndex.from_documents(documents)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=3)

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_resume(request: QueryRequest):
    prompt = f"""
You are an assistant that answers questions strictly based on the user's resume.
Only respond using the information available in the resume.
If the information is not available, say "The information is not available in the resume."

Question: {request.query}
Answer:"""

    response = query_engine.query(prompt)
    return {
        "query": request.query,
        "response": str(response).strip()
    }



from fastapi.responses import FileResponse

@app.get("/")
def home():
    return FileResponse("index.html")
