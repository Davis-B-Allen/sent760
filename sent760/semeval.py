import pandas as pd
from sklearn.preprocessing import LabelEncoder
import datetime
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import sklearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from gensim.models import KeyedVectors
import os

# Load the data
file_prefix = os.path.join(os.path.dirname(__file__), '../data/semeval/processed/')

fn1 = "twitter-2016test-A.txt"
fn2 = "aggregate.txt"
fn3 = "test.txt"

# I was having an issue where if I tried to read the twitter-2016test-A.txt
# with the quoting, the numpy array I was getting back was flawed and couldn't
# be used in the encoding
# So, I'm just treating that separately as a workaround (reading without the quoting param)
# and then concatenating everything later
data1 = pd.read_csv(file_prefix+fn1, header = None, delimiter="\t")
data2 = pd.read_csv(file_prefix+fn2, header = None, delimiter="\t", quoting=3)
data3 = pd.read_csv(file_prefix+fn3, header = None, delimiter="\t", quoting=3)

d1 = data1.values
d2 = data2.values
d3 = data3.values

# We will set the number of test samples from the size of the test.txt data
nb_validation_samples = int(d3.shape[0])

tw1 = d1[:,2]
tw2 = d2[:,2]
tw3 = d3[:,2]

c1 = d1[:,1]
c2 = d2[:,1]
c3 = d3[:,1]

print(np.unique(c1))
print(np.unique(c2))
print(np.unique(c3))

# tweets_array is the array of tweet data, class_array_raw is the array of labels
tweets_array = np.concatenate((tw1,tw2,tw3), axis=0)
class_array_raw = np.concatenate((c1,c2,c3), axis=0)

# Convert to numerical values that our models can use
encoder = LabelEncoder()
encoder.fit(class_array_raw)
class_array = encoder.transform(class_array_raw)
deep_class_array = np_utils.to_categorical(class_array)

# Prep the data for the Naive Bayes classifiers

# Take the raw tweet data and convert it to word-counts
print(datetime.datetime.now())
print("Tokenizing and counting")
# Create matrix of size num_reviews x num_tokens with counts for each word
vectorizer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
tweets_word_counts = vectorizer.fit_transform(tweets_array)
print(datetime.datetime.now())

# Create TF IDF matrix from the word counts
tfidf_transformer = TfidfTransformer()
tweets_words_tfidf = tfidf_transformer.fit_transform(tweets_word_counts)


# Prep the data for the deep learning models
max_length = 140

# tokenize the data and turn each tweet into a sequence of word indices
print(datetime.datetime.now())
print("Tokenizing and indexing")
# only use the top 20000 most commonly occurring words
tokenizer = Tokenizer(nb_words=20000)
# tokenize the reviews and generate reviews that are lists of indices for tokens in the dictionary
tokenizer.fit_on_texts(tweets_array)
sequences = tokenizer.texts_to_sequences(tweets_array)
# the dictionary of tokens to indices
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=max_length)


# Split the data based on our pre-determined number of test samples (12,284)
deep_x_train = data[:-nb_validation_samples]
deep_x_test = data[-nb_validation_samples:]

counts_x_train = tweets_word_counts[:-nb_validation_samples]
counts_x_test = tweets_word_counts[-nb_validation_samples:]

tfidf_x_train = tweets_words_tfidf[:-nb_validation_samples]
tfidf_x_test = tweets_words_tfidf[-nb_validation_samples:]

y_train = class_array[:-nb_validation_samples]
y_test = class_array[-nb_validation_samples:]

deep_y_train = deep_class_array[:-nb_validation_samples]
deep_y_test = deep_class_array[-nb_validation_samples:]

# Train a Naive Bayes classifier with the counts data
print("\n\n")
print("Training NB classifier with counts")
nb_model1 = MultinomialNB().fit(counts_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred1 = nb_model1.predict(counts_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred1)*100))
print("Recall:")
recall = recall_score(y_test, y_pred1, average='macro')
print(recall)
print("\n\n")

# Train a Naive Bayes classifier with the TF-IDF data
print("Training NB classifier with TF-IDF")
nb_model2 = MultinomialNB().fit(tfidf_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred2 = nb_model1.predict(tfidf_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred2)*100))
print("Recall:")
recall = recall_score(y_test, y_pred2, average='macro')
print(recall)
print("\n\n")

# Build an LSTM Network with no initial embedding weights
print(datetime.datetime.now())
print("Building LSTM with no initial embedding vector weights")
print("Vector length is 200")
embedding_vector_length = 200
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=max_length))
model1.add(LSTM(100))
model1.add(Dense(3, activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())
model1.fit(deep_x_train, deep_y_train, nb_epoch=3, batch_size=64)
scores1 = model1.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
pred1 = model1.predict(deep_x_test, batch_size=32, verbose=1)
predicted1 = np.argmax(pred1, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted1, average='macro')
print(recall)
print("\n\n")

# Load the GloVe twitter vectors
print(datetime.datetime.now())
print("Loading GloVe pretrained word vectors")
print("This may take some time")
vector_file_prefix = os.path.join(os.path.dirname(__file__), '../data/semeval/')
vector_file_name = "glove.twitter.27B.200d.w2vformat.txt"
en_model = KeyedVectors.load_word2vec_format(vector_file_prefix + vector_file_name)

# Initialize embedding matrix with all zeros,
# and then populate it with the Glove vectors for all words found in the vocabulary of the tweets
# If a word in the tweet vocabulary is not found in the pre-trained embeddings vocabulary,
# we just leave it as zeros
embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in word_index.items():
    if word in en_model:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = en_model[word]

# Build an LSTM Network and use the pre-trained embeddings from GloVe,
# but keep them static during training
print(datetime.datetime.now())
print("Building LSTM using fasttext pretrained vectors")
print("Vector length is 200")
print("We are keeping the word vectors static and not allowing them to be modified during training")
model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_length, trainable=False))
model2.add(LSTM(100))
model2.add(Dense(3, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model2.summary())
model2.fit(deep_x_train, deep_y_train, nb_epoch=3, batch_size=64)
scores2 = model2.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
pred2 = model2.predict(deep_x_test, batch_size=32, verbose=1)
predicted2 = np.argmax(pred2, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted2, average='macro')
print(recall)
print("\n\n")

# Build an LSTM Network and use the pre-trained embeddings from GloVe,
# but allow them to be modified during training
print(datetime.datetime.now())
print("Building LSTM using fasttext pretrained vectors")
print("Vector length is 200")
print("Word vectors can be modified during training")
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_length))
model3.add(LSTM(100))
model3.add(Dense(3, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
model3.fit(deep_x_train, deep_y_train, nb_epoch=3, batch_size=64)
scores3 = model3.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
pred3 = model3.predict(deep_x_test, batch_size=32, verbose=1)
predicted3 = np.argmax(pred3, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted3, average='macro')
print(recall)
print("\n\n")

# Build an bi-directional LSTM Network and use the pre-trained embeddings from GloVe,
# and allow them to be modified during training
print(datetime.datetime.now())
embedding_vector_length = 200
print("Building BIDIRECTIONAL LSTM using fasttext pretrained vectors")
print("Vector length is 200")
print("Word vectors can be modified during training")
model4 = Sequential()
model4.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_length))
# model4.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=max_length))
model4.add(Bidirectional(LSTM(100), merge_mode='concat'))
model4.add(Dense(3, activation='softmax'))
model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model4.summary())
model4.fit(deep_x_train, deep_y_train, nb_epoch=3, batch_size=64)
scores4 = model4.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores4[1]*100))
pred4 = model4.predict(deep_x_test, batch_size=32, verbose=1)
predicted4 = np.argmax(pred4, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted4, average='macro')
print(recall)
print("\n\n")
