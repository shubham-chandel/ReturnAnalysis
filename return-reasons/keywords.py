# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-08-11 12:58:00
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-11 15:11:34

from rakemod import Rake

import pandas as pd
from collections import defaultdict

def keywordsToIdx(idxToKeywords):
	reverse_map = defaultdict(list)
	for i, item in idxToKeywords.iteritems():
		for itm in item:
			reverse_map[itm].append(i)
	return pd.Series(reverse_map)


def generate_keywords(data_frame, minChar, maxWords, minFrequency):
	rakeObj = Rake("SmartStoplist.txt", minChar, maxWords, minFrequency)
	rakeObj.run(data_frame.return_comments)
	
	keywords = pd.Series([keyword for keyword, _ in rakeObj.keywords])
	return keywords, pd.Series(rakeObj.idxToKeywords), keywordsToIdx(pd.Series(rakeObj.idxToKeywords))


