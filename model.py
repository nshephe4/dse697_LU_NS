from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings


# Load vector store
VECTOR_STORE_PATH = "vector_store"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Load LLaMA 3.1 8B
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Prompt template
prompt = PromptTemplate.from_template("""
You are a policy assistant chatbot. Use the provided documents to answer the user's question.

Context:
{context}

Question:
{input}
""")

# Build the GRAG pipeline
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
qa_chain = create_retrieval_chain(retriever=retriever, combine_documents_chain=document_chain)

# Main interaction loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke({"input": query})
    print("\nAnswer:", result["answer"])
