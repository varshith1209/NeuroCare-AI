import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pickle
import json
import tensorflow as tf
import streamlit as st
import random

# Initialize NLTK's LancasterStemmer
stemmer = LancasterStemmer()

# Load intents file
with open("C:\\Users\\5620\\Downloads\\-MindfulMate-main\\intents1.json", encoding='utf-8') as file:
    data = json.load(file)

# Load preprocessed data
with open("C:\\Users\\5620\\Downloads\\-MindfulMate-main\\data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Load pre-trained model
model = tf.keras.models.load_model("C:\\Users\\5620\\OneDrive\\Desktop\\vs code\\bhaii\\model.h5", compile=False)  # Load the model in HDF5 format without compiling

# Function to preprocess user input and get bot response
def chatbot_response(user_input):
    user_input_processed = np.array([bag_of_words(user_input, words)])
    results = model.predict(user_input_processed)
    results_index = np.argmax(results)
    tag = labels[results_index]
    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']
            return random.choice(responses)

# Function to create bag of words from input sentence
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return bag  # Return Python list instead of np.array

# Streamlit app interface
st.title("Depression Therapist")  # Change the title

# Container to display chat history
chat_history = st.empty()

# User input text box
user_input = st.text_input("You: ")

# Initialize or retrieve previous conversations from session state
if 'previous_conversations' not in st.session_state:
    st.session_state.previous_conversations = []

previous_conversations = st.session_state.previous_conversations

# If user inputs something
if user_input:
    # Save user input to history
    previous_conversations.append({"user": user_input})
    
    # Get bot response
    bot_response = chatbot_response(user_input)
    
    if bot_response:
        # Save bot response to history
        previous_conversations.append({"bot": bot_response})
    
    # Clear user input after processing
    user_input = ""
    
    # Update session state
    st.session_state.previous_conversations = previous_conversations
    
    # Display chat history
    chat_history.markdown("<br>".join([f"**You:** {conv.get('user', '')}\n**Bot:** {conv.get('bot', '')}" for conv in previous_conversations]), unsafe_allow_html=True)
