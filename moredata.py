import pandas as pd
import random
from transformers import pipeline

# Load the original dataset
data = pd.read_csv("D:/Code/PROJECTS/Chatbot/chatbot_data.csv")

# Initialize a paraphrase pipeline
paraphrase_pipeline = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def generate_paraphrases(question, num_paraphrases=3):
    """Generate paraphrases for a given question using a pre-trained model."""
    paraphrases = []
    for _ in range(num_paraphrases):
        paraphrase = paraphrase_pipeline(question, num_return_sequences=1, num_beams=5)[0]['generated_text']
        paraphrases.append(paraphrase)
    return paraphrases

def synonym_replacement(question, synonym_dict):
    """Replace words in the question with their synonyms."""
    words = question.split()
    new_question = " ".join([synonym_dict.get(word, word) for word in words])
    return new_question

# Example synonym dictionary
synonym_dict = {
    "what": "which",
    "offer": "provide",
    "used": "utilized",
    "deliver": "ship",
    "prices": "costs",
    "products": "items",
    "order": "purchase",
    "support": "assistance",
    "contact": "reach out",
}

# Generate new data
new_data = []

for index, row in data.iterrows():
    question = row['question']
    answer = row['answer']
    
    # Add the original question-answer pair
    new_data.append((question, answer))
    
    # Generate paraphrases
    paraphrases = generate_paraphrases(question)
    for paraphrase in paraphrases:
        new_data.append((paraphrase, answer))
    
    # Apply synonym replacement
    synonym_question = synonym_replacement(question, synonym_dict)
    new_data.append((synonym_question, answer))

# Create a new DataFrame with the generated data
new_data_df = pd.DataFrame(new_data, columns=['question', 'answer'])

# Save the new data to a CSV file
new_data_df.to_csv("D:/Code/PROJECTS/Chatbot/expanded_chatbot_data.csv", index=False)

print("Generated data saved to expanded_chatbot_data.csv")
