# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-19 13:17:20
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-07-19 16:19:15

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from pickle import load, dump
from pprint import pprint

keywords = pickle.load('keywords', 'rb')

reasons = []
with open('return_reason_headset.csv') as f:
	for line in f:
		reasons.append(line)

for reason in reasons:
	print(reason)
	pprint(process.extract(reason, keywords))
	print()

