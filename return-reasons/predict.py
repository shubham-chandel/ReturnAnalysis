# -*- coding: utf-8 -*-
# @Author: Shubham Chandel
# @Date:   2016-08-18 14:02:15
# @Last Modified by:   Shubham Chandel
# @Last Modified time: 2016-08-18 14:16:56

from collections import Counter
from fuzzywuzzy import process

from helper import STOP

def predict(comment, keywords, keywordsToIdx, data_frame):
	# Remove stop words and tokenize
	commentTokenized = [word.lower() for word in comment.split() if not word.lower() in STOP]
	commentString = ' '.join(commentTokenized)
	
	# Extract most relevant keyword
	keyword = process.extractOne(commentString, keywords)[0]
	
	# Find all comments with same keyword
	indexes = keywordsToIdx[keyword]
	df = data_frame.return_sub_reason.loc[indexes]
	
	# Most used return reason
	predictedReason = Counter(df).most_common()[0][0]
	return predictedReason


# Counter(data_frame.return_sub_reason[keywordsToIdx[process.extractOne(' '.join([word.lower() for word in sent.split() if not word.lower() in STOP]), keywords)[0]]]).most_common()[0][0]

