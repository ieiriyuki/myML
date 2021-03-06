#!/usr/bin/python

"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
#clf.fit(features, labels)
#print clf.score(features, labels)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(
  features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

print "accuracy of train: ", clf.score(features_train, labels_train)
print "accuracy of test: ", clf.score(features_test, labels_test)

pred = clf.predict(features_test)

print len(labels_test)
print labels_test
print pred

print pred * labels_test 

from sklearn import metrics as mtr
print "accuracy: ", mtr.accuracy_score(labels_test, pred)
print "precision: ", mtr.precision_score(labels_test, pred)
print "recall: ", mtr.recall_score(labels_test, pred)

