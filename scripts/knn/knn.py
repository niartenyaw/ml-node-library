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

NUM_TO_COMPLETE = 1000 # 1 to 10,000 for one batch, 50,000 for all.
NUM_COLORS = 3
IMG_WIDTH = 32
NUM_BATCHES = 1 # 1 to 5
FOLDER_PRE = '../../data'
DATA_SOURCE = '/cifar-10-batches-py/'
CLASS_NAMES = pr.unpickle(FOLDER_PRE + DATA_SOURCE + 'batches.meta')['label_names']


def get_knn_classes(X_train,Y_train, X_test, k):
    results = np.empty(NUM_TO_COMPLETE)
    for i in range(NUM_TO_COMPLETE): #range(X_test.shape[0]):
    	print "Image:", i
        total_diffs = np.dot(abs(np.subtract(X_train, X_test[i])), 
        	np.ones(X_train.shape[1]))
        # Sort total_diffs and get the indices of the k smallest values
        indices = np.argsort(total_diffs)[:k]
        # Get the corresponding Y_train values. Take the mode of those.
        results[i] = np.argmax(np.bincount(Y_train[indices]))
    return results

def knn(X_train, Y_train, X_test, Y_test, k):
	results = get_knn_classes(X_train,Y_train, X_test, k).astype(int)
	np.savetxt("./results/" + str(datetime.now())[:-7] + " k_" + str(k) + " knn array.csv", results, delimiter = ',', fmt = '%1d')
	test_diff = np.subtract(results,Y_test[:NUM_TO_COMPLETE])
	num_incorrect = np.count_nonzero(test_diff)
	test_diff[test_diff > 0] = 1
	test_diff[test_diff < 0] = 1
	# correct classifications in this array are 0. Add 1 to all class labels (now 1 to 10) to differentiate them from correct answers (0)
	incorrect_classifications = np.bincount(test_diff * (Y_test[:NUM_TO_COMPLETE] + 1))[1:]
	
	overall_perc_correct = 1 - float(num_incorrect) / len(results)
	return (1 - incorrect_classifications.astype(float) / np.bincount(Y_test[:NUM_TO_COMPLETE])), overall_perc_correct

def produce_average_pixel_images(X_train, Y_train, CLASS_NAMES):
	for i in np.unique(Y_train):
		print "Producing image: ", i
		indices = np.where(Y_train == i)[0]
		l = len(indices)
		# Take the average of each pixel value across all images of the same class
		avg = (np.dot(X_train[indices].T, np.ones(l)) / l).astype(int)
		pixels = np.array(np.split(avg, NUM_COLORS)).T
		pic = np.array(np.split(pixels, IMG_WIDTH))
		im = Image.fromarray(np.uint8(pic))

		im.save("./avg_imgs/" + str(i) + '_' + CLASS_NAMES[i] + ".png")


def concat_datasets(filename, X_data=np.array([]), Y_data=np.array([])):
	batch = pr.unpickle(filename)
	if len(X_data) == 0:
		return np.array(batch['data']).astype(int), np.array(batch['labels']).astype(int)
	else:
		return np.concatenate((X_data, np.array(batch['data']).astype(int)), axis = 0), np.concatenate((Y_data, np.array(batch['labels']).astype(int)), axis = 0)

if __name__== "__main__":
	
	#fake_results = [0,1,2,3,4,5,6,7,8,9]
	#print_results_file(fake_results, 0.5, 10)

	X_train = np.array([])
	Y_train = np.array([])
	for i in range(1, NUM_BATCHES+1):
		X_train, Y_train = concat_datasets(FOLDER_PRE + DATA_SOURCE + 'data_batch_' + str(i), X_train, Y_train)
	X_test, Y_test = concat_datasets(FOLDER_PRE + DATA_SOURCE + 'test_batch')
	
	#produce_average_pixel_images(X_train, Y_train, CLASS_NAMES)
	for k in range(10,11):
		print "Working on:", k
		results, pc = knn(X_train, Y_train, X_test, Y_test, k)
		print_results_file(results, pc, k, method, DATA_SOURCE)
		

		

