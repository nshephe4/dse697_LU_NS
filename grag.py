import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Settings
DATA_FOLDER = "docs"
VECTOR_STORE_PATH = "vector_store"

# Load documents
docs = []
for file in os.listdir(DATA_FOLDER):
    file_path = os.path.join(DATA_FOLDER, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        continue
    docs.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store
vector_store = FAISS.from_documents(chunks, embedding=embeddings)
vector_store.save_local(VECTOR_STORE_PATH)

print(f"✅ Vector store saved to: {VECTOR_STORE_PATH} with {len(chunks)} chunks.")
