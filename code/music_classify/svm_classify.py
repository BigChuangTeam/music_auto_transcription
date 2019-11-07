#coding = utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    import numpy
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

#load data
x_train = np.load("mfcc\\X_train.npy")
y_train = np.load("mfcc\\Y_train.npy")
x_test = np.load("mfcc\\X_test.npy")
y_test = np.load("mfcc\\Y_test.npy")
print('Training data shape: ', x_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)

#shuffle
x_train, y_train = shuffle_in_unison(x_train, y_train)
x_test, y_test = shuffle_in_unison(x_test, y_test)

#svm
clf = svm.SVC(C=0.8, kernel='linear', decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
print("trainning set accuracy: %.6f"%clf.score(x_train, y_train))
print("testing set accuracy: %.6f"%clf.score(x_test, y_test))
