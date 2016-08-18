# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-04 14:48:11
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-17 23:46:49

import pandas as pd
import numpy as np

import nltk

from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model

import pickle
import matplotlib.pyplot as plt
from pprint import pprint

import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def evaluate_prediction(predictions, target, title="Confusion matrix"):
	print('accuracy %s' % accuracy_score(target, predictions))


def predict(vectorizer, classifier, data):
	data_features = vectorizer.transform(data.return_comments)
	predictions = classifier.predict(data_features)
	target = data.return_sub_reason
	return accuracy_score(target, predictions)


def classify(vectorizer, train_data_features, train_data, test_data):
	logreg = linear_model.LogisticRegression(n_jobs=-1, C=1e5)
	logreg = logreg.fit(train_data_features, train_data.return_sub_reason)
	print(predict(vectorizer, logreg, test_data))


def w2v_tokenize_text(text):
	tokens = []
	for sent in nltk.sent_tokenize(text):
		for word in nltk.word_tokenize(sent):
			if len(word) < 2:
				continue
			tokens.append(word)
	return tokens

def word_averaging(wv, words):
	all_words, mean = set(), []
	
	for word in words:
		if isinstance(word, np.ndarray):
			mean.append(word)
		elif word in wv.vocab:
			mean.append(wv.syn0norm[wv.vocab[word].index])
			all_words.add(wv.vocab[word].index)
		else:
			print("Word {} not in vocab.".format(word))
	
	if not mean:
		return np.zeros(wv.layer1_size,)
	
	mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
	return mean

def  word_averaging_list(wv, text_list):
	return np.vstack([word_averaging(wv, review) for review in text_list ])


# Load dataset
FOLDER = 'pickle/'
MODEL = FOLDER + 'model'
DF = FOLDER + 'data_frame'
DFUNIQ = 'pickle/data_frame_uniq_eng'

data_frame = pickle.load(open(DFUNIQ, 'rb'))

df = data_frame[data_frame.return_comments.apply(lambda x: len(x.split()) < 5)]
df = df[df.return_sub_reason.isin([i[0] for i in Counter(df.return_sub_reason).most_common() if i[1] > 500 and i[0] != 'others'][1:])]

df = data_frame.sample(1000)
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)


# 1. Bag-of-Words (BOW)
count_vectorizer = CountVectorizer(analyzer="word", tokenizer=nltk.word_tokenize, preprocessor=None, stop_words='english', max_features=3000) 
train_data_features = count_vectorizer.fit_transform(train_data.return_comments)
# print(count_vectorizer.get_feature_names())
classify(count_vectorizer, train_data_features, train_data, test_data)

# 2. n-Gram
n_gram_vectorizer = CountVectorizer(analyzer="char", ngram_range=([5,5]), tokenizer=None, preprocessor=None,  stop_words='english', max_features=3000)
train_data_features = n_gram_vectorizer.fit_transform(train_data.return_comments)
classify(n_gram_vectorizer, train_data_features, train_data, test_data)

# 3. tf-idf
tf_vect = TfidfVectorizer(min_df=2, tokenizer=nltk.word_tokenize, preprocessor=None, stop_words='english')
train_data_features = tf_vect.fit_transform(train_data.return_comments)
classify(tf_vect, train_data_features, train_data, test_data)


# 4. Word2Vec
wv = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
wv.init_sims(replace=True)

train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r.return_comments), axis=1).values
test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r.return_comments), axis=1).values

# Word averaging
X_train_word_average = word_averaging_list(wv,train_tokenized)
X_test_word_average = word_averaging_list(wv,test_tokenized)

# KNNeighbors
knn_naive_dv = KNeighborsClassifier(n_neighbors=3, n_jobs=1, algorithm='brute', metric='cosine' )
knn_naive_dv.fit(X_train_word_average, train_data.return_sub_reason)

predicted = knn_naive_dv.predict(X_test_word_average)
target = test_data.return_sub_reason
accuracy_score(target, predicted)

# Logistic Regression
logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(X_train_word_average, train_data.return_sub_reason)

predicted = logreg.predict(X_test_word_average)
target = test_data.return_sub_reason
accuracy_score(target, predicted)


# Doc2Vec
train_tagged = train_data.apply(
	lambda r: TaggedDocument(words=w2v_tokenize_text(r.return_comments), tags=[r.return_sub_reason]), axis=1)

test_tagged = test_data.apply(
	lambda r: TaggedDocument(words=w2v_tokenize_text(r.return_comments), tags=[r.return_sub_reason]), axis=1)

trainsent = train_tagged.values
testsent = test_tagged.values

doc2vec_model = Doc2Vec(trainsent, workers=2, size=100, iter=20, dm=1)

train_targets, train_regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in trainsent])
test_targets, test_regressors = zip(
	*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in testsent])

logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg = logreg.fit(train_regressors, train_targets)
evaluate_prediction(logreg.predict(test_regressors), test_targets, title=str(doc2vec_model))

knn_test_predictions = [
	doc2vec_model.docvecs.most_similar([pred_vec], topn=1)[0][0]
	for pred_vec in test_regressors
]

evaluate_prediction(knn_test_predictions, test_targets, str(doc2vec_model))





# Word Mover's Distance

train_tokenized = train_data.apply(lambda r: w2v_tokenize_text(r.return_comments), axis=1).values
test_tokenized = test_data.apply(lambda r: w2v_tokenize_text(r.return_comments), axis=1).values

flat_train_tokenized = [item for sublist in train_tokenized for item in sublist]
flat_test_tokenized = [item for sublist in test_tokenized for item in sublist]

vect = CountVectorizer(stop_words="english").fit(flat_train_tokenized)
common = [word for word in vect.get_feature_names() if word in wv]
W_common = wv[common]

vect = CountVectorizer(vocabulary=common, dtype=np.double)
X_train = vect.fit_transform(train_data.return_comments)
X_test = vect.transform(test_data.return_comments)

knn = WordMoversKNN(n_neighbors=1,W_embed=W_common, verbose=5, n_jobs=7)
knn.fit(X_train, train_data.return_sub_reason)


# Doc2Vec

from documentToVector import documentToVector
from sklearn.preprocessing import LabelEncoder

train, test = train_test_split(range(df.shape[0]))
traindf = df.iloc[train]
testdf = df.iloc[test]

model = documentToVector(df.return_comments)[0]

train_data = model.docvecs.doctag_syn0[train]
test_data = model.docvecs.doctag_syn0[test]

le = LabelEncoder()
le.fit(df.return_sub_reason)

train_targets = le.transform(traindf.return_sub_reason)
test_targets = le.transform(testdf.return_sub_reason)




fp = np.memmap("embed.dat", dtype=np.double, mode='w+', shape=wv.syn0norm.shape)
fp[:] = wv.syn0norm[:]
with open("embed.vocab", "w") as f:
	for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
		print(w, file=f)
del fp, wv

W = np.memmap("embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("embed.vocab") as f:
	vocab_list = map(str.strip, f.readlines())
vocab_dict = {w: k for k, w in enumerate(vocab_list)}



le = LabelEncoder()
le.fit(df.return_sub_reason)

docs, y = df.return_comments, le.transform(df.return_sub_reason)
docs_train, docs_test, y_train, y_test = train_test_split(docs, y,train_size=100,test_size=300,random_state=0)

vect = CountVectorizer(stop_words="english").fit(docs_train.values.tolist() + docs_test.values.tolist())
common = [word for word in vect.get_feature_names() if word in vocab_dict]
W_common = W[[vocab_dict[w] for w in common]]

vect = CountVectorizer(vocabulary=common, dtype=np.double)
X_train = vect.fit_transform(docs_train)
X_test = vect.transform(docs_test)

knn_cv = word_movers_knn.WordMoversKNNCV(cv=3, n_neighbors_try=range(1, 20), W_embed=W_common, verbose=5, n_jobs=3)
knn_cv.fit(X_train, y_train)








