# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-15 11:10:47
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-04 17:03:23

from pprint import pprint
from string import punctuation
from random import shuffle

import numpy as np
import pandas as pd

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models import Phrases

from nltk import word_tokenize, pos_tag

file = 'return_10.csv'
translator = str.maketrans({key: None for key in punctuation})

def normalize_text(line):
	return [word.lower() for word in line.translate(translator).split()]

def labeled_line_sentence(line):
	return LabeledSentence(line.return_comments, [line['index']])

data_frame = pd.read_csv(file, header=0, sep='\t', error_bad_lines=False)
data_frame = data_frame.dropna(subset = ['return_comments'])
data_frame.return_comments = data_frame.return_comments.apply(normalize_text)
data_frame = data_frame.reset_index()

sentences = data_frame.return_comments
bigram_transformer = Phrases(sentences)
model = Word2Vec(bigram_transformer[sentences], size=100, min_count=1)

# tagged_sentences = data_frame[['return_comments', 'index']].apply(labeled_line_sentence, axis=1)
# model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
# model.build_vocab(tagged_sentences)

# for epoch in range(10):
# 	shuffle(tagged_sentences)
# 	model.train(tagged_sentences)

