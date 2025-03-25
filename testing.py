from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer, pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

app = Flask(__name__)

# Define the base path for the project
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

# Performance Evaluation: Compare predictions and actual responses (dummy implementation)
def evaluate_performance(test_data, test_labels):
    correct_predictions = 0
    total_samples = len(test_data)
    
    for i, user_input in enumerate(test_data):
        user_embedding = get_embeddings([user_input])
        similarities = cosine_similarity(user_embedding, question_embeddings)
        most_similar_idx = similarities.argmax()
        predicted_answer = data['answer'].iloc[most_similar_idx]
        if predicted_answer == test_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples * 100
    print(f"Model accuracy: {accuracy:.2f}%")
    return accuracy

# Graph for Performance Evaluation (Accuracy vs Dataset Size)
def plot_performance_evaluation(test_data_sizes, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data_sizes, accuracies, marker='o', color='b', label='Accuracy')
    plt.title('Performance Evaluation: Accuracy vs Dataset')
    plt.xlabel('Test Dataset Size')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Visualization of Embeddings with t-SNE
def visualize_embeddings():
    # Reduce dimensions of embeddings using t-SNE for visualization
    embeddings_2d = TSNE(n_components=2).fit_transform(question_embeddings.detach().cpu().numpy())
    plt.figure(figsize=(10, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', s=50, cmap='viridis')
    for i, label in enumerate(predefined_questions):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    plt.title('t-SNE Visualization of Question Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

if __name__ == "__main__":
    # Example Performance Evaluation Test
    test_questions = [
        "What are your products?",
        "Can I get a discount?",
        "Where is your location?",
        "How do I contact support?",
        "Is shipping international?"
    ]
    test_labels = [
        "We offer a variety of Gujarati snacks including fafda, thepla, and khakhra.",
        "Yes, we offer discounts on bulk orders and during festive seasons.",
        "We are based in Ahmedabad, Gujarat.",
        "You can contact our customer support via email at support@example.com or call us at +91-1234567890.",
        "Currently, we only ship within India."
    ]
    
    accuracy = evaluate_performance(test_questions, test_labels)
    
    # Plot performance evaluation (example using different dataset sizes)
    test_data_sizes = [10, 20, 30, 40, 50]
    accuracies = [87.5, 90.0, 92.5, 93.0, 95.0]  # Example accuracy values
    plot_performance_evaluation(test_data_sizes, accuracies)
    
    # Visualize Embeddings with t-SNE
    visualize_embeddings()

    app.run(debug=True)
