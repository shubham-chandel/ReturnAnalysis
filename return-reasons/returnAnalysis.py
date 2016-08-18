# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-11 12:14:30
# @Last Modified by:   Shubham Chandel
# @Last Modified time: 2016-08-18 14:15:15

import pickle
from pprint import pprint

from helper import *
from keywords import generate_keywords
from cluster import cluster_documents
from incorrect import incorrect_reasons
from predict import predict

DFUNIQE = '/Users/shubham.chandel/Documents/ReturnAnalysis/pickle/data_frame_uniq_eng'
DFCSV = '/Users/shubham.chandel/Documents/ReturnAnalysis/data/return_all.csv'

class ReturnAnalysis:
	"""Generate best set of return reasons from customer’s return comment"""
	
	def __init__(self, vertical="Handsets"):
		self.vertical = vertical
		# self.loadData()
	
	def loadCluster(self, file):
		extension = file.split('.')[-1]
		
		if extension == 'pk':
			self.clusters = pickle.load(open(file, 'rb'))
		elif extension == 'txt':
			self.clusters = pd.Series(eval(open(file).read().strip()))
	
	def loadData(self, file=None):
		self.file = file
		
		# If no file, then load pickled data
		if self.file is None:
			self.data_frame = pickle.load(open(DFUNIQE, 'rb'))
		else:
			self.data_frame = pd.read_csv(self.file, header=0, sep='\t', error_bad_lines=False)
			self.data_frame = self.data_frame.groupby('return_product_category_name').get_group(self.vertical)
			self.data_frame = data_cleanup(self.data_frame, 1)
	
	def generateKeywords(self, minChar=3, maxWords=3, minFrequency=15):
		self.minChar = minChar
		self.maxWords = maxWords
		self.minFrequency = minFrequency
		
		self.keywords, self.idxToKeywords, self.keywordsToIdx = generate_keywords(self.data_frame, self.minChar, self.maxWords, self.minFrequency)
	
	def clusterKeywords(self, algorithm='doc2vec', eps=None, min_samples=None, file=None):
		
		if file is not None:
			self.loadCluster(file)
			return
		
		self.algorithm = algorithm
		self.clusters, self.documentmatrix = cluster_documents(self.keywords, self.algorithm, eps, min_samples)
	
	def incorrectAnnoated(self):
		self.incorrectdf = incorrect_reasons(self.clusters, self.data_frame, self.keywordsToIdx)
		pprint(self.incorrectdf[['return_comments', 'return_predicted_reason']].values.tolist(), open('incorrect-annoated.txt', 'w'))
	
	def incorrectAnnoatedCluster(self, clusterID):
		self.incorrectdf, self.reason = incorrect_reasons(self.clusters, self.data_frame, self.keywordsToIdx, clusterID=clusterID)
	
	def predictReturn(self, comment):
		print(predict(comment, self.keywords, self.keywordsToIdx, self.data_frame))
	
	def save(self, obj, name):
		pickle.dump(obj, open(name+'.pk', 'wb'))


def main():
	obj = ReturnAnalysis('Handsets')
	obj.loadData()
	obj.generateKeywords()
	obj.clusterKeywords()
	return obj

if __name__ == '__main__':
	main()




