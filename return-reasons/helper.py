# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-14 11:59:06
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-02 11:57:50
# @Description: Module - Language analysis helper functions

from string import punctuation
from random import shuffle

from textblob import TextBlob

from nltk.corpus import words
from nltk import word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

VOCAB = set(w.lower() for w in words.words())
STOP = set(stopwords.words('english'))
translator = str.maketrans({key: None for key in punctuation})

def part_of_speech(text):
	return  pos_tag(word_tokenize(text))

def spell_correct(text):
	return TextBlob(text).correct().raw

def sentiment(review, _type=0):
	if _type == 0:
		return  True if TextBlob(review).sentiment.polarity >= 0 else False
	elif _type == 1:
		return  TextBlob(review).sentiment.polarity
	else:  
		print("Invalid type.\nEnter 0 or 1.")

def stem_text(text):
	return ' '.join(map(SnowballStemmer("english").stem, word_tokenize(text)))

def stem_word(word):
	return SnowballStemmer("english").stem(word)

def english_word(word):
	return word.lower() in VOCAB

def clean_text(text):
	"""Return text with english words"""
	return ' '.join(word for word in word_tokenize(text) if english_word(word))

def is_english_sentence(line):
	for word in word_tokenize(line):
		if (len(word) > 2) and not english_word(word):
			return False
	return True

def normalize_text(line):
	return ' '.join(word for word in line.translate(translator).lower().split() if len(word) > 2)

def data_cleanup(data_frame, _type=0):
	"""Cleanup Dataframe"""
	
	data_frame = data_frame.dropna(subset = ['return_comments'])
	data_frame.return_comments = data_frame.return_comments.apply(normalize_text)
	
	if _type == 1:
		data_frame = data_frame.loc[data_frame.return_comments.apply(is_english_sentence)]
	
	data_frame = data_frame.reset_index()
	return data_frame


