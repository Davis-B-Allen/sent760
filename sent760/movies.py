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

import datetime
import os


# fix random seed for reproducibility
np.random.seed(7)

max_words = 20000

# moviedirtrain = r'/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/ImdbLargeMovieData/aclImdb/train'
# moviedirtest = r'/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/ImdbLargeMovieData/aclImdb/test'
moviedirtrain = os.path.join(os.path.dirname(__file__), '../data/movies/aclImdb/train')
moviedirtest = os.path.join(os.path.dirname(__file__), '../data/movies/aclImdb/test')

print(datetime.datetime.now())

print("Loading IMDB data")
movie_train = load_files(moviedirtrain, shuffle=True)
movie_test = load_files(moviedirtest, shuffle=True)
print(datetime.datetime.now())

reviews_list = movie_train.data + movie_test.data
class_array = np.concatenate((movie_train.target,movie_test.target), axis=0)
# reviews_list = movie_train.data
# class_array = movie_train.target






print(datetime.datetime.now())

print("Tokenizing and counting")
# Create matrix of size num_reviews x num_tokens with counts for each word
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. 82.2% acc.
movie_counts = movie_vec.fit_transform(reviews_list)

# Create TF IDF matrix
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)




print(datetime.datetime.now())

print("Tokenizing and indexing")
# only use the top 20000 most commonly occurring words
tokenizer = Tokenizer(nb_words=20000)
# tokenize the reviews and generate reviews that are lists of indices for tokens in the dictionary
tokenizer.fit_on_texts(reviews_list)
sequences = tokenizer.texts_to_sequences(reviews_list)
# the dictionary of tokens to indices
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=500)
labels = class_array


# nb_validation_samples = 25000
nb_validation_samples = int(0.15 * data.shape[0])

deep_x_train = data[:-nb_validation_samples]
deep_x_test = data[-nb_validation_samples:]

counts_x_train = movie_counts[:-nb_validation_samples]
counts_x_test = movie_counts[-nb_validation_samples:]

tfidf_x_train = movie_tfidf[:-nb_validation_samples]
tfidf_x_test = movie_tfidf[-nb_validation_samples:]

y_train = labels[:-nb_validation_samples]
y_test = labels[-nb_validation_samples:]




print(datetime.datetime.now())

print("\n\n")

print("Training NB classifier with counts")
nb_model1 = MultinomialNB().fit(counts_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred1 = nb_model1.predict(counts_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred1)*100))
print("\n\n")

print(datetime.datetime.now())

print("Training NB classifier with TF-IDF")
nb_model2 = MultinomialNB().fit(tfidf_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred2 = nb_model1.predict(tfidf_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred2)*100))
print("\n\n")



# print(datetime.datetime.now())
#
# embedding_vector_length = 300
# print("Building Bi-Directional LSTM using fasttext pretrained vectors")
# print("Vector length is 300")
# print("Word vectors can be modified during training")
# model4 = Sequential()
# # model4.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=500))
# model4.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=500))
# model4.add(Bidirectional(LSTM(100), merge_mode='concat'))
# model4.add(Dense(1, activation='sigmoid'))
# model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model4.summary())
# model4.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
# # Final evaluation of the model
# scores4 = model4.evaluate(deep_x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores4[1]*100))
# print("\n\n")





# print(datetime.datetime.now())
#
# print("Building LSTM with no initial embedding vector weights")
# print("Vector length is 32")
# embedding_vector_length = 32
# model0 = Sequential()
# model0.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=500))
# model0.add(LSTM(100))
# model0.add(Dense(1, activation='sigmoid'))
# model0.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model0.summary())
# model0.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
# # Final evaluation of the model
# scores0 = model0.evaluate(deep_x_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores0[1]*100))
# print("\n\n")
#
#
#
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
# Final evaluation of the model
scores1 = model1.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
print("\n\n")





print(datetime.datetime.now())

print("Loading fasttext pretrained word vectors")
print("This may take some time")
from gensim.models import FastText
from gensim.models import KeyedVectors
# en_model = FastText.load_fasttext_format('/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/fasttextVectors/wiki.en/wiki.en')
# en_model = KeyedVectors.load_word2vec_format('/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/fasttextVectors/wiki-news-300d-1M.vec')

vector_file_prefix = os.path.join(os.path.dirname(__file__), '../data/movies/')
vector_file_name = "wiki-news-300d-1M.vec"
en_model = KeyedVectors.load_word2vec_format(vector_file_prefix + vector_file_name)

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    if word in en_model:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = en_model[word]


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
# Final evaluation of the model
scores2 = model2.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
print("\n\n")

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
# Final evaluation of the model
scores3 = model3.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
print("\n\n")



print(datetime.datetime.now())

embedding_vector_length = 300
print("Building Bi-Directional LSTM using fasttext pretrained vectors")
print("Vector length is 300")
print("Word vectors can be modified during training")
model4 = Sequential()
model4.add(Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=500))
# model4.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=500))
model4.add(Bidirectional(LSTM(100), merge_mode='concat'))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model4.summary())
model4.fit(deep_x_train, y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores4 = model4.evaluate(deep_x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores4[1]*100))
print("\n\n")
