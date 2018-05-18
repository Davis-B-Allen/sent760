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

import os
dir = os.path.dirname(__file__)
file_prefix = os.path.join(os.path.dirname(__file__), '../data/semeval/processed/')


# file_prefix = "/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/SemEval2017/processed/"

fn1 = "twitter-2016test-A.txt"
fn2 = "aggregate.txt"
fn3 = "test.txt"

data1 = pd.read_csv(file_prefix+fn1, header = None, delimiter="\t")
data2 = pd.read_csv(file_prefix+fn2, header = None, delimiter="\t", quoting=3)
data3 = pd.read_csv(file_prefix+fn3, header = None, delimiter="\t", quoting=3)

d1 = data1.values
d2 = data2.values
d3 = data3.values

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

tweets_array = np.concatenate((tw1,tw2,tw3), axis=0)
class_array_raw = np.concatenate((c1,c2,c3), axis=0)

encoder = LabelEncoder()
encoder.fit(class_array_raw)
class_array = encoder.transform(class_array_raw)
deep_class_array = np_utils.to_categorical(class_array)





print(datetime.datetime.now())
print("Tokenizing and counting")
# Create matrix of size num_reviews x num_tokens with counts for each word
vectorizer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)
tweets_word_counts = vectorizer.fit_transform(tweets_array)
print(datetime.datetime.now())

# Create TF IDF matrix
tfidf_transformer = TfidfTransformer()
tweets_words_tfidf = tfidf_transformer.fit_transform(tweets_word_counts)




# max_length = 67
max_length = 140


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





deep_x_train = data[:-nb_validation_samples]
deep_x_test = data[-nb_validation_samples:]

counts_x_train = tweets_word_counts[:-nb_validation_samples]
counts_x_test = tweets_word_counts[-nb_validation_samples:]

tfidf_x_train = tweets_words_tfidf[:-nb_validation_samples]
tfidf_x_test = tweets_words_tfidf[-nb_validation_samples:]

y_train = class_array[:-nb_validation_samples]
y_test = class_array[-nb_validation_samples:]

# deep_y_train and deep_y_test
deep_y_train = deep_class_array[:-nb_validation_samples]
deep_y_test = deep_class_array[-nb_validation_samples:]





print("\n\n")

print("Training NB classifier with counts")
nb_model1 = MultinomialNB().fit(counts_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred1 = nb_model1.predict(counts_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred1)*100))
# print("Report:\n")
# report = classification_report(y_test, y_pred1)
# print(report)
print("Recall:")
recall = recall_score(y_test, y_pred1, average='macro')
print(recall)



print("\n\n")


print("Training NB classifier with TF-IDF")
nb_model2 = MultinomialNB().fit(tfidf_x_train, y_train)
# Predicting the Test set results, find accuracy
y_pred2 = nb_model1.predict(tfidf_x_test)
print("Accuracy: %.2f%%" % (sklearn.metrics.accuracy_score(y_test, y_pred2)*100))
# print("Report:\n")
# report = classification_report(y_test, y_pred1)
# print(report)
print("Recall:")
recall = recall_score(y_test, y_pred2, average='macro')
print(recall)

print("\n\n")













# deep_y_train and deep_y_test

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
# Final evaluation of the model
scores1 = model1.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores1[1]*100))
print("\n\n")
pred1 = model1.predict(deep_x_test, batch_size=32, verbose=1)
predicted1 = np.argmax(pred1, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted1, average='macro')
print(recall)





print(datetime.datetime.now())

print("Loading GloVe pretrained word vectors")
print("This may take some time")
from gensim.models import KeyedVectors
# en_model = KeyedVectors.load_word2vec_format('/Users/davisallen/Education/Hunter/Classes/CSCI 760 - Computer Linguistics/Projects/SentiProject/Data/GloveTwitterVectors/glove.twitter.27B.200d.w2vformat.txt')
vector_file_prefix = os.path.join(os.path.dirname(__file__), '../data/semeval/')
vector_file_name = "glove.twitter.27B.200d.w2vformat.txt"
en_model = KeyedVectors.load_word2vec_format(vector_file_prefix + vector_file_name)





embedding_matrix = np.zeros((len(word_index) + 1, 200))
for word, i in word_index.items():
    if word in en_model:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = en_model[word]



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
# Final evaluation of the model
scores2 = model2.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores2[1]*100))
print("\n\n")
pred2 = model2.predict(deep_x_test, batch_size=32, verbose=1)
predicted2 = np.argmax(pred2, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted2, average='macro')
print(recall)






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
# Final evaluation of the model
scores3 = model3.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
print("\n\n")
pred3 = model3.predict(deep_x_test, batch_size=32, verbose=1)
predicted3 = np.argmax(pred3, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted3, average='macro')
print(recall)






print(datetime.datetime.now())

embedding_vector_length = 200
print("Building BIDIRECTIONAL LSTM using fasttext pretrained vectors")
print("Vector length is 200")
print("Word vectors can be modified during training")
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1, 200, weights=[embedding_matrix], input_length=max_length))
# model3.add(Embedding(len(word_index) + 1, embedding_vector_length, input_length=max_length))
model3.add(Bidirectional(LSTM(100), merge_mode='concat'))
model3.add(Dense(3, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model3.summary())
model3.fit(deep_x_train, deep_y_train, nb_epoch=3, batch_size=64)
# Final evaluation of the model
scores3 = model3.evaluate(deep_x_test, deep_y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores3[1]*100))
print("\n\n")
pred3 = model3.predict(deep_x_test, batch_size=32, verbose=1)
predicted3 = np.argmax(pred3, axis=1)
print("Recall:")
recall = recall_score(y_test, predicted3, average='macro')
print(recall)






# End
