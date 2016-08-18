# -*- coding: utf-8 -*-
# @Author: shubham.chandel
# @Date:   2016-07-08 11:08:34
# @Last Modified by:   shubham.chandel
# @Last Modified time: 2016-08-11 12:44:59

from rakemod import Rake
import pandas as pd

from helper import *

import pickle
from pprint import pprint

file = 'return_all.csv'

FOLDER = 'headset_all/'
MODEL = FOLDER + 'model'
DF = FOLDER + 'data_frame'
DFUNIQ = FOLDER + 'data_frame_uniq'
DFUNIQE = FOLDER + 'data_frame_uniq_eng'

data_frame = pickle.load(open(DFUNIQE, 'rb'))

data_frame = pd.read_csv(file, header=0, sep='\t', error_bad_lines=False)
data_frame = data_frame.groupby('return_product_category_name').get_group('Handsets')
data_frame = data_cleanup(data_frame, 1)

rake_object = Rake("SmartStoplist.txt", 3, 3, 20)
keywords = rake_object.run(data_frame.return_comments)

# pickle.dump(keywords, open( "keywords.pk", "wb" ) )

