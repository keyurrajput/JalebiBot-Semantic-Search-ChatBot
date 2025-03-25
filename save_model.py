import os
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# Specify the model name (you can change this to any model you want to use)
model_name = "distilbert-base-uncased"

# Load the model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the base path for your project
base_path = "D:/Code/PROJECTS/Chatbot/"
model_path = os.path.join(base_path, "chatbot_model")

# Save the model and tokenizer
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Model and tokenizer saved to", model_path)
