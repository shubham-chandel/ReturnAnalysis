# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-11 14:27:54
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-11 15:14:26

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from multiprocessing import cpu_count
import numpy as np

CORES = cpu_count()

class LabeledLineSentence:
	def __init__(self, sentences):
		self.sentences = []
		for idx, line in enumerate(sentences):
			self.sentences.append(LabeledSentence(words=line.split(), tags=[idx]))
	
	def __iter__(self):
		for item in self.sentences:
			yield item
	
	def sentences_perm(self):
		np.random.shuffle(self.sentences)


def documentToVector(documents):
	obj = LabeledLineSentence(documents)
	
	simple_models = [
		Doc2Vec(dm=0, size=300, negative=5, hs=0, min_count=1, workers=CORES),
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

