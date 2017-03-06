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
    info['datetime'] = str(datetime.now())[:-7].replace(':','-')
    info['pc_by_class'] = ','.join(info['pc_by_class'].astype('|S10'))
    info['computer'] = ' '.join(info['computer']).translate(None, ',/:')
    
    #Write detailed file 
    filename = './../../data' + info['data_source'] + 'results/' + \
                info['method'] + ' ' + str(info['num_images']) + ' ' + info['datetime'] + '.txt'
    outfile = open(filename, 'w')
    outfile.write('Results\n')
    outfile.write('\n')
    outfile.write('Method:' + info['method'] + '\n')
    outfile.write('Data source:' + info['data_source'] + '\n')
    outfile.write('datetime:' + info['datetime'] + '\n')
    outfile.write('Number of images included:' + str(info['num_images']) + '\n')
    outfile.write('Computer:' + str(info['computer']) + '\n')
    outfile.write('Time:' + str(info['time']) + '\n')
    outfile.write('\n')
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

    ## add results to summary file too
    filename = './../../data' + info['data_source'] + 'results_summary.csv'
    outfile = open(filename, 'a')
    del info['other']
    del info['class_names']
    outfile.write(','.join([str(info[i]) for i in sorted(info.iterkeys())]) + '\n')
    outfile.close()
    