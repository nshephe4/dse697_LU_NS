from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct")
pipe(messages)

query = "Explain the difference between grace and mercy from a biblical perspective."

# Generate output
output = pipe(query, max_new_tokens=200, do_sample=True, temperature=0.7)

# Print the response
print(output[0]["generated_text"])