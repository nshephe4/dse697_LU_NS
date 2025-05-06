from transformers import AutoTokenizer, AutoModelForCausalLM

# Load LLaMA 3.1 8B model and tokenizer
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Initialize conversation history
chat_history = []

# Main interaction loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        print("👋 Exiting the chatbot. Goodbye!")
        break

    # Prepare the input prompt
    history_str = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history])
    prompt = f"{history_str}\nUser: {query}\nAssistant:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        repetition_penalty=1.2,
        top_p=0.95,
        do_sample=True
    )

    # Decode output (skip input tokens)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()

    print("\nAnswer:", answer)

    # Update conversation history
    chat_history.append((query, answer))
