# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-02 15:54:35
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-03 12:12:45

import os
from gensim import corpora, models, utils

from collections import defaultdict, Counter

def topic_prob_extractor(hdp=None, topn=None):
	topic_list = hdp.show_topics(topics=-1, topn=topn)
	topics = [int(x.split(':')[0].split(' ')[1]) for x in topic_list]
	split_list = [x.split(' ') for x in topic_list]
	weights = []
	for lst in split_list:
		sub_list = []
		for entry in lst: 
			if '*' in entry: 
				sub_list.append(float(entry.split('*')[0]))
		weights.append(np.asarray(sub_list))
	sums = [np.sum(x) for x in weights]
	return pd.DataFrame({'topic_id' : topics, 'weight' : sums})

class ReutersCorpus():
	def __init__(self, documents):
		self.tokens = documents.apply(utils.simple_preprocess)
		self.dictionary = corpora.Dictionary(self.tokens)
		self.dictionary.filter_extremes()
	
	def __iter__(self):
		for token in self.tokens:
			yield self.dictionary.doc2bow(token)


corpus = ReutersCorpus(data_frame.return_comments)

mallet_path = 'mallet/bin/mallet'
model = models.wrappers.LdaMallet(mallet_path, corpus, num_topics=30, id2word=corpus.dictionary)
# model = models.LdaMulticore(corpus, num_topics=20, id2word=corpus.dictionary)
model = models.HdpModel(list(corpus), id2word=corpus.dictionary)

x = defaultdict(list)
for idx, comment in data_frame.return_comments.iteritems():
	topic = model[corpus.dictionary.doc2bow(utils.simple_preprocess(data_frame.return_comments[idx]))]
	if len(topic) > 0:
		x[topic[0][0]].append(comment)


