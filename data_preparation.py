import json
import pickle
import numpy as np
import random
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load intents from the JSON file
def load_intents(json_file='intents.json'):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['intents']

# Load the model and preprocessing objects
model = load_model('model.h5')

with open('texts.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('responses.pkl', 'rb') as f:
    responses = pickle.load(f)

# Function to preprocess the text input
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# Function to get the chatbot's response
# Function to get the chatbot's response
def get_response(user_input):
    # Load the saved labels (classes)
    with open('labels.pkl', 'rb') as file:
        classes = pickle.load(file)

    # Continue with the rest of the function

    processed_input = preprocess(user_input)
    bag = [0] * len(vectorizer)  # vectorizer is actually the list of words

    for word in processed_input.split():
        if word in vectorizer:
            index = vectorizer.index(word)
            bag[index] = 1

    prediction = model.predict(np.array([bag]))[0]
    max_index = np.argmax(prediction)
    intent = classes[max_index]

    
    return random.choice(responses.get(intent, ["I'm here to listen. How can I help you today?"]))


