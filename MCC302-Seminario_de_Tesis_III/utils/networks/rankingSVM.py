"""
Implementation of pairwise ranking using scikit-learn LinearSVC
Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.
Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
"""

import itertools
import numpy as np

from sklearn import svm, linear_model
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold


def transform_pairwise(X, y):
  """Transforms data into pairs with balanced labels for ranking
  Transforms a n-class ranking problem into a two-class classification
  problem. Subclasses implementing particular strategies for choosing
  pairs should override this method.
  In this method, all pairs are choosen, except for those that have the
  same target value. The output is an array of balanced classes, i.e.
  there are the same number of -1 as +1
  Parameters
  ----------
  X : array, shape (n_samples, n_features)
      The data
  y : array, shape (n_samples,) or (n_samples, 2)
      Target labels. If it's a 2D array, the second column represents
      the grouping of samples, i.e., samples with different groups will
      not be considered.
  Returns
  -------
  X_trans : array, shape (k, n_feaures)
      Data as pairs
  y_trans : array, shape (k,)
      Output class labels, where classes have values {-1, +1}
  """
  X_new = []
  y_new = []
  y = np.asarray(y)
  if y.ndim == 1:
    y = np.c_[y, np.ones(y.shape[0])]
  comb = itertools.combinations(range(X.shape[0]), 2)
  for k, (i, j) in enumerate(comb):
    if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
      # skip if same target or different group
      continue
    X_new.append(X[i] - X[j])
    y_new.append(np.sign(y[i, 0] - y[j, 0]))
    # output balanced classes
    if y_new[-1] != (-1) ** k:
      y_new[-1] = - y_new[-1]
      X_new[-1] = - X_new[-1]
  return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
  """Performs pairwise ranking with an underlying LinearSVC model
  Input should be a n-class ranking problem, this object will convert it
  into a two-class classification problem, a setting known as
  `pairwise ranking`.
  See object :ref:`svm.LinearSVC` for a full description of parameters.
  """

  def fit(self, X, y):
    """
    Fit a pairwise ranking model.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,) or (n_samples, 2)
    Returns
    -------
    self
    """
    X_trans, y_trans = transform_pairwise(X, y)
    super(RankSVM, self).fit(X_trans, y_trans)
    return self

  def predict(self, X):
    """
    Predict an ordering on X. For a list of n samples, this method
    returns a list from 0 to n-1 with the relative order of the rows of X.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
    Returns
    -------
    ord : array, shape (n_samples,)
        Returns a list of integers representing the relative order of
        the rows in X.
    """
    if hasattr(self, 'coef_'):
      #np.argsort(np.dot(X, self.coef_.T))
      return np.argsort(np.dot(X, self.coef_.T).ravel())
    else:
      raise ValueError("Must call fit() prior to predict()")

  def score(self, X, y):
    """
    Because we transformed into a pairwise problem, chance level is at 0.5
    """
    X_trans, y_trans = transform_pairwise(X, y)
    return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


if __name__ == '__main__':
  # as showcase, we will create some non-linear data
  # and print the performance of ranking vs linear regression

  np.random.seed(1)
  n_samples, n_features = 300, 5
  true_coef = np.random.randn(n_features)
  X = np.random.randn(n_samples, n_features)
  noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
  y = np.dot(X, true_coef)
  y = np.arctan(y) # add non-linearities
  y += .1 * noise  # add noise
  Y = np.c_[y, np.mod(np.arange(n_samples), 5)]  # add query fake id
  #cv = KFold(n_samples, 5)
  cv = KFold(5)
  print(cv.split(X))

  # make a simple plot out of it
  import pylab as pl
  pl.scatter(np.dot(X, true_coef), y)
  pl.title('Data to be learned')
  pl.xlabel('<X, coef>')
  pl.ylabel('y')
  pl.show()

  for train, test in cv.split(X):
    # print the performance of ranking
    rank_svm = RankSVM().fit(X[train], Y[train])
    print('Performance of ranking ', rank_svm.score(X[test], Y[test]))

    # and that of linear regression
    ridge = linear_model.RidgeCV(fit_intercept=True)
    ridge.fit(X[train], y[train])
    X_test_trans, y_test_trans = transform_pairwise(X[test], y[test])
    score = np.mean(np.sign(np.dot(X_test_trans, ridge.coef_)) == y_test_trans)
    print('Performance of linear regression ', score)
