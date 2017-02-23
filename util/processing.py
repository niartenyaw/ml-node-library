import numpy as np
from datetime import datetime
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
def print_results_file(info):
	dt = str(datetime.now())
	filename = 'results/' + dt[:-7] + '.txt'
	outfile = open(filename, 'w')
	outfile.write('Results\n')
	outfile.write('\n')
	outfile.write('Method:' + info['method'] + '\n')
	outfile.write('Data source:' + info['data_source'] + '\n')
	outfile.write('datetime:' + dt + '\n')
	outfile.write('Number of images included:' + str(info['num_images']) + '\n')
	outfile.write('\n')
	outfile.write('Overall accuracy:' + str(info['pc_overall']) + '\n')
	outfile.write('\n')
	outfile.write('Accuracy by class:\n')
	for i in range(len(info['class_names'])):
		outfile.write(info['class_names'][i] + ':' + str(info['pc_by_class'][i]) + '\n')
	outfile.write('\n')

	for key in info['other'].keys():
		outfile.write(key + ':' + str(info['other'][key]) + '\n')
	outfile.close()