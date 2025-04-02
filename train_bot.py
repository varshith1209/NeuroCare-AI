import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# Initialize NLTK's LancasterStemmer
stemmer = LancasterStemmer()

# Load intents file
with open("C:\\Users\\5620\\Downloads\\-MindfulMate-main\\intents1.json", encoding='utf-8') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

# Preprocess data
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        bag.append(1) if w in wrds else bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("C:\\Users\\5620\\Downloads\\-MindfulMate-main\\data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

model = Sequential([
    Dense(8, input_shape=(len(training[0]),), activation='relu'),
    Dense(6, activation='relu'),
    Dense(len(output[0]), activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Debugging information
current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

# Save model in HDF5 format
model_path = os.path.join(current_directory, "model.h5")
model.save(model_path)
print(f"Model saved at: {model_path}")
