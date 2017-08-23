#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC

clf = SVC(C=10000.,kernel='rbf')

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

st = time()
clf.fit(features_train,labels_train)
print "fitting time:", round(time() - st,3), "s"

st = time()
#score = clf.score(features_test,labels_test)
#print "prediction time:", round(time() - st,3), "s"

#print score
prediction = clf.predict(features_test)
print prediction[10], prediction[26], prediction[50]

print sum(prediction)

#########################################################


