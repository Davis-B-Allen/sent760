from sklearn.datasets import load_files
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from gensim.models import FastText
from gensim.models import KeyedVectors
import datetime
import os

# fix random seed for reproducibility
np.random.seed(7)

# Load data
moviedirtrain = os.path.join(os.path.dirname(__file__), '../data/movies/aclImdb/train')
moviedirtest = os.path.join(os.path.dirname(__file__), '../data/movies/aclImdb/test')

print(datetime.datetime.now())
print("Loading IMDB data")
movie_train = load_files(moviedirtrain, shuffle=True)
movie_test = load_files(moviedirtest, shuffle=True)
print(datetime.datetime.now())

# The authors of the data divided the data up into a 50/50 train test split
# but we want to use more of the data for training
# So, we'll concatenate the provided data into a big 50,000 item array
# and split it later based on a different train/test split ratio
reviews_list = movie_train.data + movie_test.data
class_array = np.concatenate((movie_train.target,movie_test.target), axis=0)

# Prep the data for the Naive Bayes classifiers

# Take the review data and convert it to word-counts
print(datetime.datetime.now())
print("Tokenizing and counting")
# Create matrix of size num_reviews x num_tokens with counts for each word
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
movie_counts = movie_vec.fit_transform(reviews_list)

# Create TF IDF matrix from the word counts
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

# Prep the data for the deep learning models

# tokenize the data and turn each review into a sequence of word indices
print(datetime.datetime.now())
print("Tokenizing and indexing")
tokenizer = Tokenizer(nb_words=20000)
# tokenize the reviews and generate reviews that are lists of indices for tokens in the dictionary
tokenizer.fit_on_texts(reviews_list)
sequences = tokenizer.texts_to_sequences(reviews_list)
# the dictionary of tokens to indices
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=500)
labels = class_array

# Choose the number of test samples for the train/test split
# We're going to use an 85/15 train/test split
nb_validation_samples = int(0.15 * data.shape[0])

deep_x_train = data[:-nb_validation_samples]
deep_x_test = data[-nb_validation_samples:]

counts_x_train = movie_counts[:-nb_validation_samples]
counts_x_test = movie_counts[-nb_validation_samples:]

tfidf_x_train = movie_tfidf[:-nb_validation_samples]
tfidf_x_test = movie_tfidf[-nb_validation_samples:]

y_train = labels[:-nb_validation_samples]
y_test = labels[-nb_validation_samples:]

# Train a Naive Bayes classifier with the counts data
print(datetime.datetime.now())
print("\n\n")
print("Training NB classifier with counts")
nb_model1 = MultinomialNB().fit(counts_x_train, y_train)
y_pred1 = nb_model1.predict(counts_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred1)*100))
print("\n\n")

# Train a Naive Bayes classifier with the TF-IDF data
print(datetime.datetime.now())
print("Training NB classifier with TF-IDF")
nb_model2 = MultinomialNB().fit(tfidf_x_train, y_train)
y_pred2 = nb_model1.predict(tfidf_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred2)*100))
print("\n\n")

# Build an LSTM Network with no initial embedding weights
print(datetime.datetime.now())
print("Building LSTM with no initial embedding vector weights")
print("Vector length is 300")
embedding_vector_length = 300
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=500))
model1.add(LSTM(100))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
model1.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
scores1 = model1.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
print("\n\n")

# Load the Fasttext vectors
print(datetime.datetime.now())
print("Loading fasttext pretrained word vectors")
print("This may take some time")
vector_file_prefix = os.path.join(os.path.dirname(__file__), '../data/movies/')
vector_file_name = "wiki-news-300d-1M.vec"
en_model = KeyedVectors.load_word2vec_format(vector_file_prefix + vector_file_name)

# Initialize embedding matrix with all zeros,
# and then populate it with the fasttext vectors for all words found in the vocabulary of the movie reviews
# If a word in the movie review vocabulary is not found in the pre-trained embeddings vocabulary,
# we just leave it as zeros
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    if word in en_model:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = en_model[word]

# Build an LSTM Network and use the pre-trained embeddings from Fasttext,
# but keep them static during training
print(datetime.datetime.now())
print("Building LSTM using fasttext pretrained vectors")
print("Vector length is 300")
print("We are keeping the word vectors static and not allowing them to be modified during training")
model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=500, trainable=False))
model2.add(LSTM(100))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
model2.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
scores2 = model2.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
print("\n\n")

# Build an LSTM Network and use the pre-trained embeddings from Fasttext,
# but allow them to be modified during training
print(datetime.datetime.now())
print("Building LSTM using fasttext pretrained vectors")
print("Vector length is 300")
print("Word vectors can be modified during training")
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=500))
model3.add(LSTM(100))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
model3.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
scores3 = model3.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
print("\n\n")
