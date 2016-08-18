# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-18 11:35:29
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-17 17:57:50

from pprint import pprint
from string import punctuation
from random import shuffle
import multiprocessing

import numpy as np
import pandas as pd

from helper import *

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from sklearn.cluster import DBSCAN
from sklearn import cluster, datasets

file = 'return_10.csv'
translator = str.maketrans({key: None for key in punctuation})
cores = multiprocessing.cpu_count()

# 
class LabeledLineSentence():
	def __init__(self, sentences):
		self.sentences = []
		for idx, line in enumerate(sentences):
			self.sentences.append(LabeledSentence(words=line.split(), tags=[idx]))
	
	def __iter__(self):
		for item in self.sentences:
			yield item
	
	def sentences_perm(self):
		np.random.shuffle(self.sentences)


def extract_features(documents):
	obj = LabeledLineSentence(documents)
	
	simple_models = [
		Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=1, workers=cores),
	]
	
	[model.build_vocab(obj.sentences) for model in simple_models]
	
	alpha, min_alpha, passes = (0.025, 0.001, 100)
	alpha_delta = (alpha - min_alpha) / passes
	
	for epoch in range(passes):
		obj.sentences_perm()
		for model in simple_models:
			model.min_alpha, model.alpha = alpha, alpha
			model.train(obj.sentences)	
		alpha -= alpha_delta
	
	return simple_models


doc_all = np.array([[(1/model[0].docvecs.similarity(i, j))-1 for i, _ in enumerate(documents)] for j , _ in enumerate(documents)])


def doc(x, y):
	i, j = int(x[0]), int(y[0])
	return doc_all[i][j]
	# return (1/model[0].docvecs.similarity(i, j))-1




# 
def cluster_vectors(vectors, n_clusters=5):
	k_means = cluster.KMeans(n_clusters=n_clusters)
	k_means.fit(vectors)
	
	clusters = [[] for _ in range(n_clusters)]
	for idx, pred in enumerate(k_means.labels_):
		clusters[pred].append(keywords[idx][0])
	
	return clusters

def preprocess(doc):
	doc = doc.lower()  # Lower the text.
	doc = word_tokenize(doc)  # Split into words.
	doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
	doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
	return doc


def top_similar(idx, models, documents):
	for idxi in range(idx):
		print('--> ', documents[idxi])
		for model in models:
			for i, weight in model.docvecs.most_similar(idxi, topn=10):
				print(i, documents[i], weight)
			print()


def most_similar_ed(model, idx):
	"""Most similar vector based on Euclidean distance"""
	return [min(enumerate(np.linalg.norm(model[0].docvecs[i]-model[0].docvecs[idx]) for i in range(len(documents)) if i != idx), key=lambda x: x[1])]


# WMD
def wmd(i, j):
	v_1, v_2 = vect.transform([documents[i], documents[j]])
	v_1 = v_1.toarray().ravel().astype(np.double)
	v_2 = v_2.toarray().ravel().astype(np.double)
	return emd(v_1, v_2, D_)

W_ = wv[[w if w in wv else w.capitalize() for w in vect.get_feature_names()]]

from sklearn.metrics import euclidean_distances
D_ = euclidean_distances(W_)
D_ = D_.astype(np.double)
D_ /= D_.max()

doc_vectors = []
for vector in vect.transform(documents):
	v = vector.toarray().ravel().astype(np.double)
	v /= v.sum()
	doc_vectors.append(v)

wmd_all = [[emd(doc_vectors[i], doc_vectors[j], D_) for i, _ in enumerate(documents)] for j , _ in enumerate(documents)]

wmd_all = np.array([[wv.wmdistance(documents_prep[i], documents_prep[j]) for i, _ in enumerate(documents_prep)] for j , _ in enumerate(documents_prep)])

instance = WmdSimilarity(documents, wv, num_best=10)


def most_similar_wmd(idx):
	return sorted(enumerate(wmd_all[idx]), key=lambda x: x[1])[:10]


def wmd(x, y):
	i, j = int(x[0]), int(y[0])
	return wmd_all[i][j]
	# return wv.wmdistance(documents[i], documents[j])

# 
data_frame = pd.read_csv(file, header=0, sep='\t', error_bad_lines=False)
data_frame = data_frame.groupby('return_product_category_name').get_group('Handsets')
# data_frame = data_frame.groupby('return_reason').get_group('ACCESSORY_DEFECTIVE')
data_frame = data_cleanup(data_frame, 0)


# 
FOLDER = 'headset_all/'
KW = 'keywords-4'

keywords = pickle.load(open(FOLDER + KW, 'rb'))
documents = pd.Series([' '.join(filter(lambda z: z.lower() not in STOP, x.split())).lower() for x in open('return_reason_headset.csv')] + [x[0] for x in keywords])
documents = data_frame.return_comments.unique()
documents_prep = documents.apply(preprocess)

model = extract_features(documents)
top_similar(50, model, documents)


# Cluster vectors: My
visited = [False for _ in range(len(documents))]
clusters = []
for idx, document in enumerate(documents):
	if not visited[idx]:
		cluster = {document}
		for j, weight in model[0].docvecs.most_similar(idx, topn=10):
			if weight >= 0.9:
				if not visited[j]:
					visited[j] = True
					cluster.add(documents[j])
			else: break
		if len(cluster) > 1:
			clusters.append([idx, cluster])


# Cluster vectors: DFS
THRESHOLD = 0.65
DISTANCE = 'wmd'
visited = set()
for u in range(len(documents)):
	if u not in visited:
		stack = [u]
		lvisited = set()
		while stack:
			vertex = stack.pop()
			if vertex not in visited:
				visited.add(vertex)
				lvisited.add(vertex)
				if DISTANCE == 'cosine':
					stack.extend(set(x for x,w in model[0].docvecs.most_similar(int(vertex)) if w>=THRESHOLD) - visited)
				elif DISTANCE == 'euclidean':
					stack.extend(set(x for x,w in most_similar_ed(model, int(vertex)) if w>=THRESHOLD) - visited)
				elif DISTANCE == 'wmd':
					stack.extend(set(x for x,w in instance[documents[vertex]] if w>=THRESHOLD) - visited)
					# stack.extend(set(x for x,w in most_similar_wmd(int(vertex)) if w<=THRESHOLD) - visited)
				else: print('Something miserable happened !')
		cluster = [documents[i] for i in lvisited]
		if len(cluster) > 0:
			print(cluster)




# Cluster vectors: DBSCAN
# clusters = cluster_vectors(model.docvecs)
db = DBSCAN(eps=0.2, min_samples=1000).fit(model.docvecs)

labels = db.labels_
print(len(set(labels)) - (1 if -1 in labels else 0))

for idx, label in enumerate(labels):
	if label == -1:
		print(documents[idx])

X = np.arange(len(documents)).reshape(-1, 1)

# Counter(DBSCAN(eps=.46, min_samples=2, metric=wmd).fit(X).labels_)
# Counter(DBSCAN(eps=.1, min_samples=1, metric=doc).fit(X).labels_)

groups = defaultdict(list)
for idx, group in enumerate(DBSCAN(eps=.09, min_samples=2, metric=objx.doc).fit(objx.X).labels_):
	groups[group].append(objx.documents[idx])


for k, v in groups.items():
	if k != -1: 
		print(k)
		pprint(v)


import time
start_time = time.time()
main()
print((time.time() - start_time)*len(documents)/60)


reverse_map = defaultdict(list)
for i, item in rake_object.phrase_map.iteritems():
	for itm in item:
		reverse_map[itm].append(i)

predictedReason = defaultdict(set)
for clusterID, keywords in obj.clusters.items():
	idxs = np.unique([i for keyword in keywords for i in obj.keywordsToIdx[keyword]])
	df = obj.data_frame.iloc[idxs]
	reasonPercentage = [num*100/df.shape[0] for reason,num in Counter(df.return_sub_reason).most_common()]
	reason = Counter(df.return_sub_reason).most_common()[0][0]
	if reasonPercentage[0] >= 50.0:
		print(df.shape[0], reasonPercentage[0])
		# for idx, item in df[df.return_sub_reason != reason].iterrows():
		for idx, item in df.iterrows():
			predictedReason[idx].add(reason)

z = obj.data_frame.join(pd.DataFrame(pd.Series(predictedReason).apply(lambda x: ' '.join(x)), columns=['return_predicted_reason']), how='right')





