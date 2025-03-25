import pandas as pd
from transformers import pipeline

# Sample initial dataset
data = {
    "question": [
        "What products do you offer?",
        "What ingredients are used in fafda?",
        "How do you deliver the products?",
        "What are the prices of your products?",
        "Do you offer discounts?",
        "Can I customize my order?",
        "Are your products gluten-free?",
        "Do you use preservatives?",
        "Where are you located?",
        "What is your return policy?",
        "How can I contact customer support?",
        "What payment methods do you accept?",
        "Do you ship internationally?",
        "Can I track my order?",
        "How can I provide feedback?"
    ],
    "answer": [
        "We offer a variety of Gujarati snacks including fafda, thepla, and khakhra.",
        "The main ingredients in fafda are gram flour, carom seeds, and turmeric.",
        "We deliver the products via local courier services within 3-5 business days.",
        "The prices of our products range from INR 50 to INR 200 per pack.",
        "Yes, we offer discounts on bulk orders and during festive seasons.",
        "Yes, you can customize your order by specifying your preferences while placing the order.",
        "Most of our products are gluten-free. Please check the product description for details.",
        "No, we do not use any artificial preservatives in our products.",
        "We are based in Ahmedabad, Gujarat.",
        "We accept returns within 7 days of delivery if the product is unopened and in its original condition.",
        "You can contact our customer support via email at support@example.com or call us at +91-1234567890.",
        "We accept all major credit/debit cards, UPI, and net banking.",
        "Currently, we only ship within India.",
        "Yes, you will receive a tracking number once your order is shipped.",
        "You can provide feedback through our website or email us at feedback@example.com."
    ]
}

# Load data into a DataFrame
df = pd.DataFrame(data)

# Initialize the paraphrase pipeline
paraphrase_pipeline = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# Function to generate paraphrases
def generate_paraphrases(question, num_paraphrases=3):
    paraphrases = []
    for _ in range(num_paraphrases):
        results = paraphrase_pipeline(f"paraphrase: {question}")[0]['generated_text']
        if results not in paraphrases:
            paraphrases.append(results)
    return paraphrases

# List to hold new rows
new_rows = []

# Loop through each question and answer pair in the data
for idx, row in df.iterrows():
    question = row['question']
    answer = row['answer']
    
    # Generate paraphrases for the question
    paraphrases = generate_paraphrases(question)
    
    # Add the original question-answer pair
    new_rows.append({'question': question, 'answer': answer})
    
    # Add the paraphrased question-answer pairs
    for paraphrased_question in paraphrases:
        new_rows.append({'question': paraphrased_question, 'answer': answer})

# Create a new DataFrame with the expanded data
expanded_data = pd.DataFrame(new_rows)

# Save the expanded data to a new CSV file
expanded_data.to_csv("expanded_chatbot_data.csv", index=False)
print("Generated data saved to expanded_chatbot_data.csv")
