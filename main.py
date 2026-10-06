import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
print("Key prefix:", gemini_api_key[:3])
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


vector_store=QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"

)
user_query="could u please tell me the password pattern? "

search_results=vector_store.similarity_search(user_query)

context = "\n\n".join(
    result.page_content
    for result in search_results
)

SYSTEM_PROMPT = f"""
Use the following context to answer the question.

Context:
{context}
"""
client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        { "role": "system", "content":SYSTEM_PROMPT  },
        { "role": "user", "content":user_query  },
    ]
)

print(f"🤖: {response.choices[0].message.content}")