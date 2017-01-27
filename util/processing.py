import numpy as np
'''
Format Pickle data into dictionary.

For CIFAR-10 data set: http://www.cs.utoronto.ca/~kriz/cifar.html
Keys
    labels: range 0-9 (see numeric meanings in cifar batches.meta)
    data: 10000x3072 array. each array row is red (top row of image down), green, blue
'''
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    datadict = cPickle.load(fo)
    fo.close()
    return datadict

'''

'''
