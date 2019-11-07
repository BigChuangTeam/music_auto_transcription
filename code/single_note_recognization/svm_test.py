
# coding: utf-8

# In[1]:



import random
import numpy as np
import matplotlib.pyplot as plt

from __future__ import print_function
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


try:
   del X_train, y_train
   del X_test, y_test
   print('Clear previously loaded data.')
except:
   pass

X_train = np.load('data\\train_x2.npy')
y_train = np.load('data\\train_y2.npy')
X_test = np.load('data\\test_x.npy')
y_test = np.load('data\\test_y.npy')

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# In[3]:


num_training = 5544
num_validation = 1584
num_test = 791

mask = range(num_test, num_test + num_validation)
X_val = X_test[mask]
y_val = y_test[mask]

mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# In[4]:


#  append the bias dimension of ones (i.e. bias trick) so that our SVM
# only has to worry about optimizing a single weight matrix W.

X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

print(X_train.shape, X_val.shape, X_test.shape)


# In[5]:


from codes.svm import LinearSVM
from codes.linear_svm import svm_loss_vectorized
import time


# In[14]:


#use validation data to find a better hyperparamater

lr = 8e-8
rs = 2.6e4

svm = LinearSVM()
loss_hist = svm.train(X_train,y_train, lr, rs, num_iters=1500)
y_train_pre = svm.predict(X_train)
train_accu = np.mean(y_train_pre == y_train)
y_val_pre = svm.predict(X_val)
val_accu = np.mean(y_val_pre == y_val )
y_test_pre = svm.predict(X_test)
test_accu = np.mean(y_test_pre == y_test )

print('train accuracy: %f' % train_accu)
print('validation accuracy: %f' % val_accu)
print('test accuracy: %f' % test_accu)

plt.plot(loss_hist)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


# In[23]:


# test some singo wav from other dataset

from codes.Features import *

path = "piano\\60.wav"    

feature = wav2feature(path)
print(feature.shape)


# In[24]:


feature = np.hstack([feature, np.ones((feature.shape[0], 1))])
pre_1 = svm.predict(feature)
print(pre_1)

