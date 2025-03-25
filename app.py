from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Define the base path for your project
base_path = "D:/Code/PROJECTS/Chatbot/"

# Load the model and tokenizer
model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the dataset
data_path = os.path.join(base_path, "expanded_chatbot_data.csv")
if not os.path.isfile(data_path):
    raise ValueError(f"Data file does not exist: {data_path}")

data = pd.read_csv(data_path)

# Generate embeddings for predefined questions
def get_embeddings(questions):
    inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

predefined_questions = data['question'].tolist()
question_embeddings = get_embeddings(predefined_questions)

@app.route('/')
def index():
    return send_from_directory(base_path, 'index.html')

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"response": "Please provide a message."}), 400

    # Generate embedding for user input
    user_embedding = get_embeddings([user_input])

    # Compute cosine similarity
    similarities = cosine_similarity(user_embedding, question_embeddings)
    most_similar_idx = similarities.argmax()
    response = data['answer'].iloc[most_similar_idx]

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True) 
    





