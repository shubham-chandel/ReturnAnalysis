# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-12 16:20:44
# @Last Modified by:   Shubham Chandel
# @Last Modified time: 2016-08-18 16:27:26

import pickle
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt

# DFPATH = '/Users/shubham.chandel/Documents/ReturnAnalysis/pickle/data_frame'
DFPATH = 'data_frame'

class PipelineError:
	"""Statistics on return-reasons not listed"""
	
	def __init__(self, vertical='Handsets'):
		self.vertical = vertical
		
		self.df = pickle.load(open(DFPATH, 'rb'))
		self.df.return_sub_reason = self.df.return_sub_reason.apply(lambda x: x.upper() if not isinstance(x, float) else x)
		
		self.data_frame = self.df.groupby('return_product_category_name').get_group(self.vertical)
	
	def changeVertical(self, vertical):
		self.vertical = vertical
		self.data_frame = self.df.groupby('return_product_category_name').get_group(self.vertical)
	
	def notListed(self):
		return set(self.data_frame.return_sub_reason.unique()) - set(x.strip().upper() for x in open(self.vertical + '.txt'))
	
	def notListedFrame(self):
		return self.data_frame[self.data_frame.return_sub_reason.isin(self.notListed())]
	
	def plot(self):
		pd.DataFrame(pd.to_datetime(self.notListedFrame().return_date_time).apply(lambda x: x.date())).groupby('return_date_time').size().plot()
		plt.title('Count non-listed reasons vs Time')
		plt.ylabel('Count')
		plt.xlabel('Time')
		plt.show()
	
	def count(self):
		self.frequency = self.notListedFrame().groupby('return_sub_reason').size().sort_values()
		pprint(self.frequency)


