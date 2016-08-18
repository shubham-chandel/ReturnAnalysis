# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-26 12:37:17
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-12 16:13:13

import pandas as pd
import pickle

from collections import Counter

FOLDER = 'headset_all/'
MODEL = FOLDER + 'model'
DF = FOLDER + 'data_frame'
DFUNIQ = FOLDER + 'data_frame_uniq'

def prominent_reason(idx):
	print(return_reason[idx], return_comments[idx])
	for itemid, dotproduct in model.most_similar(idx):
		print(return_reason[itemid], return_comments[itemid], dotproduct, itemid)

def mismatch_best():
	mismatch = {}
	for idx, row in data_frame[['return_comments', 'return_reason']].iterrows():
		comment = row.return_comments
		reason = row.return_reason
		
		reasons = [reason]
		for itemid, dotproduct in model.most_similar(int(idx), topn=10):
			if dotproduct >= 0.9:
				itemreason = return_reason[itemid]
				reasons.append(itemreason)
		
		true_reason, count = Counter(reasons).most_common(1)[0]
		if count < 3:
			continue
		
		if true_reason != reason:
			try:
				mismatch[reason].append((comment, idx, true_reason))
			except:
				mismatch[reason] = [(comment, idx, true_reason)]
		
		if idx%1000 == 0:
			print(idx) 
	
	return mismatch

model = pickle.load(open(MODEL, 'rb')).docvecs
data_frame = pickle.load(open(DFUNIQ, 'rb'))

return_comments = data_frame.return_comments
return_reason = data_frame.return_reason


for idx, row in data_frame[['return_comments', 'return_reason']].iterrows():
	comment = row.return_comments
	reason = row.return_reason
	
	reasons, items = [reason], [(idx, reason)]
	for itemid, dotproduct in model.most_similar(int(idx), topn=10):
		if dotproduct >= 0.7:
			itemreason = return_reason[itemid]
			reasons.append(itemreason)
			items.append((itemid, itemreason))
	
	true_reason, count = Counter(reasons).most_common(1)[0]
	if count > 3:
		continue
	
	for item in items:
		mismatch[item][true_reason] += 1


# pd.DataFrame(pd.to_datetime(data_frame[data_frame.return_reason.isin(set(data_frame.return_reason.unique()) - set(x.strip().upper() for x in open('return_reason_headset.csv')))].return_date_time.dropna()).apply(lambda x: x.date()).sort_values()).groupby('return_date_time').size().plot()


# data_frame[data_frame.return_sub_reason.isin(set(data_frame.return_sub_reason.unique()) - set(x.strip().upper() for x in open('return_sub_reason.txt')))].groupby('return_sub_reason').size()


