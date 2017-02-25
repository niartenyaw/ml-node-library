'''
Using the CIFAR-10 Data Set to write a k-Nearest Neighbors algorithm
32x32 - red, green, blue
'''
import sys
sys.path.insert(0, '../../util/')
import numpy as np
import matplotlib as plt
import processing as pr
from sklearn import svm
import platform
import time

FOLDER_PRE = '../../data'
DATA_SOURCE = '/cifar-10-batches-py/'
CLASS_NAMES = pr.unpickle(FOLDER_PRE + DATA_SOURCE + 'batches.meta')['label_names']

NUM_TO_COMPLETE = 10000
NUM_COLORS = 3
IMG_WIDTH = 32
NUM_BATCHES = 1 # 1 to 5

loss_options = ['hinge', 'squared_hinge']
penalty_options = ['l1', 'l2']

if __name__== "__main__":
    t0 = time.time()
    i=1
    filename = FOLDER_PRE + DATA_SOURCE + 'data_batch_' + str(i)
    b_train = pr.unpickle(filename)
    X_train = np.array(b_train['data'])[:NUM_TO_COMPLETE].astype(int)
    y_train = np.array(b_train['labels'])[:NUM_TO_COMPLETE].astype(int)

    filename = FOLDER_PRE + DATA_SOURCE + 'data_batch_' + str(i+1)
    b_test = pr.unpickle(filename)
    X_test = np.array(b_test['data'])[:NUM_TO_COMPLETE].astype(int)
    y_test = np.array(b_test['labels'])[:NUM_TO_COMPLETE].astype(int)

    lin_clf = svm.LinearSVC()
    lin_clf.fit(X_train, y_train)

    pred = lin_clf.predict(X_test)

    diff = (pred - y_test)

    incorrect = np.array([0 if i==0 else 1 for i in diff])  # 0 is correct, 1 is incorrect

    # +1 to the classes on y_test (now 1 to 10) to differentiate it from 0 (a correct prediction)
    incorrect_by_class = np.bincount(incorrect * (y_test+1))[1:]
    print incorrect_by_class
    print np.bincount(y_test)
    pc_by_class = 1 - incorrect_by_class.astype(float) / np.bincount(y_test).astype(float)
    print pc_by_class

    np.set_printoptions(threshold=np.nan)
    pr.print_results_file(dict(
            pc_overall=(1.0 - sum(incorrect).astype(float) / NUM_TO_COMPLETE),
            pc_by_class=pc_by_class,
            class_names=CLASS_NAMES,
            data_source=DATA_SOURCE,
            num_images=NUM_TO_COMPLETE,
            method='svm linear',
            computer=platform.uname(),
            time=time.time() - t0,
            other=dict(
                params=str(lin_clf),
                coefficients=lin_clf.coef_,
                intercepts=lin_clf.intercept_,
                ),
        )
    )

