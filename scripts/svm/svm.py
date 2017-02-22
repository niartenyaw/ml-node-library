'''
Using the CIFAR-10 Data Set to write a k-Nearest Neighbors algorithm
32x32 - red, green, blue
'''
import sys
sys.path.insert(0, '../../util/')
import numpy as np
import scipy.stats as stats
import matplotlib as plt
import processing as pr
from PIL import Image
from datetime import datetime
from pprint import pprint
from sklearn import svm

FOLDER_PRE = '../../data'
DATA_SOURCE = '/cifar-10-batches-py/'
CLASS_NAMES = pr.unpickle(FOLDER_PRE + DATA_SOURCE + 'batches.meta')['label_names']


NUM_COLORS = 3
IMG_WIDTH = 32
NUM_BATCHES = 1 # 1 to 5

if __name__== "__main__":
	i=1
	filename = FOLDER_PRE + DATA_SOURCE + 'data_batch_' + str(i)
	batch = pr.unpickle(filename)
	X = np.array(batch['data'])#.astype(int)
	y = np.array(batch['labels'])#.astype(int)
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(X, y)