# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-17 17:47:26
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-17 20:19:30

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

class IncorrectAnnoated():
	"""Find incorrect annoated return-comments"""
	
	def __init__(self, file=None, clusters=None):
		
		if file is not None:
			self.file = file
			self.load(file)
		else:
			self.clusters = clusters
	
	def load(self, file):
		extension = file.split('.')[-1]
		
		if extension == 'pk':
			self.clusters = pickle.load(open(file, 'rb'))
		elif extension == 'txt':
			self.clusters = pd.Series(eval(open(file).read().strip()))
	
	def incorrectAnnoated(self, data_frame, keywordsToIdx, threshold=50.0):
		
		predictedReason = defaultdict(set)
		for clusterID, keywords in self.clusters.items():
			df = data_frame.iloc[np.unique([i for keyword in keywords for i in keywordsToIdx[keyword]])]
			keywordCount = Counter(df.return_sub_reason).most_common()
			
			reasonPercentage = [count*100/df.shape[0] for keyword, count in keywordCount]
			topreasonPercentage = reasonPercentage[0]
			
			if topreasonPercentage >= threshold:
				reason = keywordCount[0][0]
				for idx in df.index:
					predictedReason[idx].add(reason)
		
		predictedReason = pd.Series(predictedReason)
		predictedReason = predictedReason.apply(lambda x: ' and '.join(x))
		self.predictedReason = pd.DataFrame(predictedReason, columns=['return_predicted_reason'])
		
		incorrectdf = data_frame.join(self.predictedReason, how='right')
		self.incorrectdf = incorrectdf[incorrectdf.return_sub_reason != incorrectdf.return_predicted_reason]

	def incorrectAnnoatedCluster(self, data_frame, keywordsToIdx, clusterID):
		keywords = self.clusters[clusterID]
		df = data_frame.iloc[np.unique([i for keyword in keywords for i in keywordsToIdx[keyword]])]
		
		keywordCount = Counter(df.return_sub_reason).most_common()
		self.reason = keywordCount[0][0]
		
		# self.incorrectclusterdf = df[df.return_sub_reason != self.reason]
		self.incorrectclusterdf = df
		

def incorrect_reasons(clusters, data_frame, keywordsToIdx, threshold=50.0, clusterID=None):
	obj = IncorrectAnnoated(clusters=clusters)
	
	if clusterID is not None:
		obj.incorrectAnnoatedCluster(data_frame, keywordsToIdx, clusterID)
		return obj.incorrectclusterdf, obj.reason
	else:
		obj.incorrectAnnoated(data_frame, keywordsToIdx, threshold)
		return obj.incorrectdf
	




