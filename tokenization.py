import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'do you think my dog is amazing?'
]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
sequence = tokenizer.texts_to_sequences(sentences)
print(sequence)
pad_sequence = pad_sequences(sequence,maxlen=5, truncating='post')
print(pad_sequence)
