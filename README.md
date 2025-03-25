# JalebiBot

**Semantic search chatbot for enhanced customer service experience**

## Overview
JalebiBot is a sophisticated customer service chatbot designed specifically for "Jalebi Eats," leveraging semantic search capabilities to deliver accurate and contextually relevant responses to customer inquiries. Built with modern natural language processing techniques, this application understands the intent behind customer questions regardless of how they are phrased, providing a seamless and intuitive communication experience.

## Technology Stack
- **Flask**: Lightweight web framework for serving the application
- **Transformers**: Hugging Face's library for state-of-the-art NLP models
- **Sentence-BERT**: Semantic embeddings for understanding query intent
- **PyTorch**: Deep learning framework for processing embeddings
- **Scikit-learn**: Used for cosine similarity calculations
- **Pandas**: Data manipulation and storage
- **HTML/CSS/JavaScript**: Frontend interface for user interaction

## Key Components

### 1. Semantic Search Engine
The heart of JalebiBot is its semantic search capability powered by the `sentence-transformers/paraphrase-MiniLM-L6-v2` model. Rather than relying on exact keyword matching, the system:
- Converts user queries into high-dimensional vector embeddings
- Compares these embeddings against a database of pre-embedded questions
- Identifies the most semantically similar questions using cosine similarity
- Returns the corresponding answers to the user

### 2. Data Augmentation
To enhance the robustness of the chatbot, the project includes data augmentation techniques:
- **Paraphrasing**: Generates alternative phrasings of questions using the T5 paraphrase model
- **Synonym Replacement**: Substitutes words with semantically equivalent alternatives
- **Dataset Expansion**: Increases the variety of question formats while maintaining answer consistency

### 3. User Interface
The chatbot features a clean, user-friendly interface with:
- Orange and white color scheme reflecting the Jalebi Eats brand
- Real-time message display with color-coded user and bot messages
- Responsive design that works across different devices
- Simple text input and send button for effortless interaction

### 4. Testing and Evaluation
The project includes comprehensive testing capabilities:
- Performance evaluation against test datasets
- Accuracy metrics for model assessment
- Visualization of embeddings using t-SNE for qualitative analysis
- Dataset size vs. accuracy plotting for quantitative insights

## Project Structure
- **app.py**: Main Flask application serving the chatbot
- **chatbot_model.py**: Initial dataset creation and paraphrasing
- **index.html**: Frontend interface for user interaction
- **moredata.py**: Advanced data augmentation with synonym replacement
- **save_model.py**: Utility for saving models locally
- **testing.py**: Performance evaluation and visualization tools

## Implementation Details
The chatbot operates through a straightforward workflow:
1. User enters a question through the web interface
2. The question is sent to the Flask backend via a POST request
3. The backend converts the question into embedding vectors
4. Cosine similarity identifies the closest matching pre-defined question
5. The corresponding answer is returned and displayed to the user

## Future Enhancements
- **Multi-turn Conversations**: Implementing context awareness for follow-up questions
- **Intent Classification**: Adding explicit intent recognition for improved accuracy
- **Sentiment Analysis**: Detecting user frustration or satisfaction
- **Feedback Loop**: Incorporating user feedback to improve responses over time
- **Multilingual Support**: Extending capabilities to handle multiple languages

## Deployment Instructions
1. Install required dependencies: `pip install flask pandas torch transformers scikit-learn`
2. Update the base path in app.py to match your project directory
3. Ensure the dataset CSV file is present in the specified location
4. Run the Flask application: `python app.py`
5. Access the chatbot through your browser at `http://localhost:5000`

JalebiBot demonstrates how modern NLP techniques can be applied to create intelligent customer service solutions that understand natural language variations and provide consistent, accurate responses.
