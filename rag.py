import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
folder=Path("docs")
file_content=[]
for file in folder.glob("*.txt"):
    loader=TextLoader(file)
    docs=loader.load()
    for doc in docs:
        file_content.append(doc)

a=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400

)
b=a.split_documents(documents=file_content)
print(b[0].page_content)
# Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=gemini_api_key
)
vector_store=QdrantVectorStore.from_documents(
    documents=b,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"

)
print(vector_store)
print("______index____-")