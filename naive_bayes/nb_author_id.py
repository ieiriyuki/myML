#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import MultinomialNB


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



#########################################################
### your code goes here ###

clf = MultinomialNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time() - t0, 3), "s"

num = 0
correct = 0
for item in pred:
    if item == labels_test[num]:
        correct += 1
    #print "pred: %d, true: %d" % (item, labels_test[num])
    num += 1

ratio = (correct*1.0) / (num*1.0)
print "correct: %d, total: %d, ratio: %f" % (correct, num, ratio)
print clf.score(features_test, labels_test)

#########################################################


