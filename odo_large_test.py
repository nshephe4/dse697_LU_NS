# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("image-text-to-text", model="meta-llama/Llama-4-Scout-17B-16E-Instruct")
pipe(messages)

query = "Explain the difference between grace and mercy from a biblical perspective."

# Generate output
output = pipe(query, max_new_tokens=200, do_sample=True, temperature=0.7)

# Print the response
print(output[0]["generated_text"])