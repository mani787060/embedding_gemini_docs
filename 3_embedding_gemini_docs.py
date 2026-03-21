import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# 2. Your specific list of documents
documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

# 3. Initialize the Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)

# 4. Create the Chunks (USING create_documents instead of split_text)
# This handles the list correctly!
doc_objects = text_splitter.create_documents(documents)

# 5. Extract the text back from the objects to embed them
chunks = [doc.page_content for doc in doc_objects]

print(f"Total items to embed: {len(chunks)}")

# 6. Generate and Print Vectors
try:
    vector_list = embeddings.embed_documents(chunks)
    
    for i, vector in enumerate(vector_list):
        print(f"Vector {i+1} (Preview): {vector[:32]}...")
        
except Exception as e:
    print(f"Error: {e}")