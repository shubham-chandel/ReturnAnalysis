# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-11 14:41:46
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-17 17:40:53

import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.cluster import DBSCAN

from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

from documentToVector import documentToVector

STOP = set(stopwords.words('english'))

def preprocess(doc):
	doc = doc.lower()  # Lower the text.
	doc = word_tokenize(doc)  # Split into words.
	doc = [w for w in doc if not w in STOP]  # Remove stopwords.
	doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
	return doc


class ClusterDocuments:
	def __init__(self, documents, algorithm='doc2vec'):
		self.documents = documents
		self.algorithm = algorithm
		
		self.X = np.arange(len(self.documents)).reshape(-1, 1)
	
	def metric(self, x, y):
		i, j = int(x[0]), int(y[0])
		return self.documentMatrix[i][j]
	
	def extractVector(self):
		if self.algorithm == 'doc2vec':
			model = documentToVector(self.documents)
			self.documentMatrix = np.array([[(1/model[0].docvecs.similarity(i, j))-1 for i, _ in enumerate(self.documents)] for j , _ in enumerate(self.documents)])
		
		elif self.algorithm == 'wmd':
			documentspre = self.documents.apply(preprocess)
			
			wv = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
			wv.init_sims(replace=True)
			
			self.documentMatrix = np.array([[wv.wmdistance(documentspre[i], documentspre[j]) for i, _ in enumerate(documentspre)] for j , _ in enumerate(documentspre)])
			
			del wv
	
	def cluster(self, eps=None, min_samples=None):
		if self.algorithm == 'doc2vec':
			if eps is None:
				eps = 0.09
			if min_samples is None:
				min_samples = 2
		
		elif self.algorithm == 'wmd':
			if eps is None:
				eps = 0.45
			if min_samples is None:
				min_samples = 2
		
		self.clusters = defaultdict(list)
		for idx, group in enumerate(DBSCAN(eps=eps, min_samples=min_samples, metric=self.metric).fit(self.X).labels_):
			self.clusters[group].append(self.documents[idx])
		
		self.clusters = pd.Series(self.clusters)
		self.clusters = self.clusters.drop(-1)


def cluster_documents(documents, algorithm='doc2vec', eps=None, min_samples=None):
	obj = ClusterDocuments(documents, algorithm)
	obj.extractVector()
	obj.cluster(eps, min_samples)
	return obj.clusters, obj.documentMatrix


