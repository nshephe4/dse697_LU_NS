from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Load vector store
VECTOR_STORE_PATH = "vector_store"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Load LLaMA 3.1 8B
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True, skip_special_tokens=True)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.1,
    return_full_text=False,
    repetition_penalty=1.2,
    top_p=0.95,
    streamer=streamer
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize conversation history
chat_history = []

# Prompt template
prompt_template = """
You are a policy assistant chatbot for the IAEA (International Atomic Energy Agency). 

Use ONLY the provided documents and the conversation history to answer the user's question.  
- Be very formal, precise, and government-appropriate.  
- Summarize the provided context before answering.
- Cite the document name, chapter, and header where the answer is derived from.
- Do NOT make up any information.
- Do NOT return or repeat the history itself in the output.
- Do NOT hallucinate sources — only cite if they were in the provided context.
- Do NOT return the prompt structure

Format your response like this:

---
Summary of Context:
<Your summarized context here>

Answer:
<Formal, direct answer here>

Source:
- Document Name: <document name>
- Chapter: <chapter title or number>
- Header: <header title or section>
---

Conversation History:
{history}

Context:
{context}

Question:
{input}
"""

prompt = PromptTemplate.from_template(prompt_template)

# Build the GRAG pipeline
document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
qa_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

# Main interaction loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("👋 Exiting the chatbot. Goodbye!")
        break

    # Create a history string
    history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])

    # Query the GRAG chain
    result = qa_chain.invoke({
        "input": query,
        "history": history_str
    })

    answer = result["answer"]
    print("\nAnswer:", answer)

    # Update conversation history
    chat_history.append((query, answer))

