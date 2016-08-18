# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-12 21:07:14
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-07-14 12:49:58

import numpy as np
import pandas as pd

from language import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

file = 'return_5.csv'

data_frame = pd.read_csv(file, header=0, sep='\t', error_bad_lines=False)
data_frame = data_frame.groupby('return_product_category_name').get_group('Books')
data_frame = data_frame[['return_comments', 'return_reason']].dropna()
data_frame = data_frame.reset_index()

data_frame.return_comments = data_frame.return_comments.apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(data_frame.return_comments, data_frame.return_reason)

text_clf = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier()),
])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))

