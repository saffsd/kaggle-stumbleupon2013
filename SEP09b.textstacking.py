"""
Use cross-validation to generate level-1 stacking features for
the text fields we have.

Marco Lui, September 2013
"""

# -*- coding: utf-8 -*-
import inspect, os
import numpy as np
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.linear_model as lm
import pandas as p
import json
from collections import defaultdict
import scipy.sparse as sp

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import roc_auc_score

def read_unk(val):
  if val == '?':
    return 0
  elif isinstance(val,basestring):
    return int(val)
  else:
    return val

from sklearn.cross_validation import StratifiedKFold
def cv_predict(rd, X, y, n_folds=10):
  """
  Generate a vector of predictions for each X using cross-validation
  """
  skf = StratifiedKFold(y, n_folds)
  retval = np.zeros_like(y, dtype=float)
  for train, test in skf:
    rd.fit(X[train], y[train])
    pred = rd.predict_proba(X[test])[:,1]
    retval[test] = pred
  return retval


def main():
  ###
  # TOOLS
  ###
  tfv = TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1)

  logr = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1.0, class_weight=None, random_state=None)
  randf = RandomForestClassifier(2000, n_jobs=-1)
  s = StandardScaler()

  ###
  # RAW DATA
  ###
  print "loading data.."
  traindata = p.read_table('../../data/train.tsv', converters={'is_news':read_unk, 'news_front_page':read_unk})
  testdata  = p.read_table('../../data/test.tsv', converters={'is_news':read_unk, 'news_front_page':read_unk})
  moredata  = p.read_table('../raw/extra.tsv')
  data = p.concat((traindata, testdata))
  data = p.merge(data[['urlid','boilerplate']], moredata)
  y = np.array(traindata['label'])

  keys_text = [ k for k in data if data.dtypes[k] == object ]

  print "KEYS", keys_text

  l0 = []
  for k in keys_text:
    print "KEY", k
    try:
      vec = tfv.fit_transform([ t if type(t) == str else "" for t in data[k]])
    except AttributeError:
      import pdb;pdb.post_mortem()
    
    train_vec = vec[:len(traindata)]
    test_vec  = vec[len(traindata):]

    train_pred = cv_predict(logr, train_vec, y)

    logr.fit(train_vec, y)
    test_pred = logr.predict_proba(test_vec)[:,1]

    pred = np.hstack((train_pred, test_pred))
    l0.append((k, pred))

  vec = np.vstack(zip(*l0)[1]).T

  print "transform data"
  X = s.fit_transform( vec )
  X_train = X[:len(traindata)]
  X_test  = X[len(traindata):]

  rd = logr
  print "predictions for {}".format(rd)
  scores = cross_validation.cross_val_score(rd, X_train, y, cv=10, scoring='roc_auc')
  print "CV Score: {0:.5f}".format(np.mean(scores))

  print "fit data"
  rd.fit(X_train,y)
  fvals = dict(zip(zip(*l0)[0], rd.coef_.ravel()))

  print "feature importance:"
  for k in sorted(fvals, key=fvals.get, reverse=True):
    print "  {0:<30}: {1:.5f}".format(k, fvals[k])

  pred = rd.predict_proba(X_test)[:,1]
  pred_df = p.DataFrame({'urlid':testdata['urlid'], 'label':pred}, columns=['urlid','label'])
  
  outpath = os.path.splitext(os.path.basename(inspect.getfile(inspect.currentframe())))[0] + '.csv'
  print "writing output to: {}".format(outpath)
  pred_df.to_csv(outpath, index=False)

if __name__=="__main__":
  main()
