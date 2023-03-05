#importing modules
import json
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
#opening the file in read mode 
with open("sarcasm.json", 'r') as f:
    dataload = json.load(f)
#creating three lists of heading, link, sentiment point
sentences = []
labels = []
urls = []
#loading the data in lists
for items in dataload:
    sentences.append(items['headline'])
    labels.append(items['is_sarcastic'])
    urls.append(items['article_link'])
# tokenizer = Tokenizer(oov_token="<OOV>")
# tokenizer.fit_on_texts(sentences)
# # print(tokenizer.word_index)
# word_index = tokenizer.word_index
# sequences = tokenizer.texts_to_sequences(sentences)
# padded = pad_sequences(sequences=sequences, padding='post')
# print(padded[0])
# print(padded.shape)

#creating training and testing datapoints
training_size = 20000
training_labels = sentences[0:training_size]
training_target = labels[0:training_size]
test_labels = sentences[training_size:]
test_target = labels[training_size:]
#creating tokenizer instance and <OOV> for tokenizing each word
tokenizer = Tokenizer(oov_token="<OOV>", num_words=1000000)
tokenizer.fit_on_texts(training_labels)
word_index = tokenizer.word_index
#sequencing words of sentences
sequences_train = tokenizer.texts_to_sequences(training_labels)
#making pads for relatively smaller sentences
padded_train = pad_sequences(sequences_train, maxlen=40, padding='post', truncating='post')
# tokenizer.fit_on_texts(test_labels)
sequences_test = tokenizer.texts_to_sequences(test_labels)
padded_test = pad_sequences(sequences_test, maxlen=40, padding='post', truncating='post')
#creating the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(tokenizer.word_counts, 300000, input_length=40),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(24, activation="relu"),
#     tf.keras.layers.Dense(1, activation = "sigmoid")
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.word_counts, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
#compilation
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
num_of_epochs = 30
history = model.fit(padded_train, training_target, batch_size=128, epoch = num_of_epochs, validation_data=[padded_test,test_target], verbose =2)

