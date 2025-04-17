# Use a pipeline as a high-level helper
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
model = AutoModelForImageTextToText.from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")

# Define your text-only query
query = "Explain the doctrine of justification by faith alone."

# Preprocess text input only (no image)
inputs = processor(text=query, return_tensors="pt").to(device)

# Generate output
generated_ids = model.generate(**inputs, max_new_tokens=256)

# Decode the result
response = processor.decode(generated_ids[0], skip_special_tokens=True)

print(response)