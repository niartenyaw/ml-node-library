'''
Using the CIFAR-10 Data Set to write a k-Nearest Neighbors algorithm
'''
import sys
sys.path.insert(0, '../../util/')
import numpy as np
import scipy.stats as stats
import matplotlib as plt
import processing as pr

def knn(X_train,Y_train, X_test, k):
    results = np.empty(100)
    for i in range(100): #range(X_test.shape[0]):
        total_diffs = np.dot(abs(np.subtract(X_train, X_test[i])), 
        	np.ones(X_train.shape[1]))
        # Sort total_diffs and get the indices of the k smallest values
        indices = np.argsort(total_diffs)[:k]
        # Get the corresponding Y_train values. Take the mode of those.
        results[i] = np.argmax(np.bincount(Y_train[indices]))
    return results

if __name__== "__main__":
	for k in range(3,11):
		print
		print "Working on:", k
		traindata = pr.unpickle('../../data/cifar-10-batches-py/data_batch_1')
		testdata = pr.unpickle('../../data/cifar-10-batches-py/test_batch')
		
		X_train = np.array(traindata['data']).astype(int)
		Y_train = np.array(traindata['labels']).astype(int)
		X_test = np.array(testdata['data']).astype(int)
		Y_test = np.array(testdata['labels']).astype(int)

		results = knn(X_train,Y_train, X_test, k).astype(int)
		np.savetxt("./results/knn_array.csv", results, delimiter = ',', fmt = '%1d')
		test_diff = np.subtract(results,Y_test[:100])
		num_incorrect = np.count_nonzero(test_diff)
		perc_correct = 1 - float(num_incorrect) / len(results)
		print "Percentage correct: ", perc_correct