import numpy as np
from random import shuffle
"""
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
"""
 


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W)  #N,C
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores_correct = scores[  np.arange(num_train), y ]  #1,N
  scores_correct = np.reshape( scores_correct,(num_train, 1))  #N,1
  margin = np.maximum(scores - scores_correct + 1.0, 0.0)
  margin[np.arange(num_train),y] = 0.0
  loss += np.sum(margin) / num_train
  loss += 0.5 * reg * np.sum(W*W)

  coeff_mat = np.zeros((num_train,num_classes))
  coeff_mat[margin>0] = 1
  coeff_mat[np.arange(num_train),y] = 0
  coeff_mat[np.arange(num_train), y] = -np.sum(coeff_mat, axis=1 )
  dW = (X.T).dot(coeff_mat)
  dW = dW / num_train + reg * W

  return loss, dW
