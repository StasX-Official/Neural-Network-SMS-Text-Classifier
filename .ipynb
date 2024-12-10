# import libraries
try:
  # %tensorflow_version only exists in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
test_data['label'] = label_encoder.transform(test_data['label'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['message'])
train_sequences = tokenizer.texts_to_sequences(train_data['message'])
test_sequences = tokenizer.texts_to_sequences(test_data['message'])
max_len = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

model = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index)+1, 16, input_length=max_len),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_sequences, train_data['label'], epochs=10, validation_data=(test_sequences, test_data['label']))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    # Preprocess the input message
    pred_seq = tokenizer.texts_to_sequences([pred_text])
    pred_seq = pad_sequences(pred_seq, maxlen=max_len, padding='post')
    prediction = model.predict(pred_seq)[0][0]
    if prediction >= 0.5:
        label = 'spam'
    else:
        label = 'ham'
    return [prediction, label]

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]
  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True
  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False
  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
