#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as plt
from my_defs import getFrac

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person'] # You will need to use more features
print("len of features are {0}".format(len(features_list)))

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop('TOTAL')

#print data_dict['METTS MARK']

features_list.append('to_frac')
features_list.append('from_frac')

na_count = {}
mean = {}
var = {}
for key in features_list[1:]:
    na_count[key] = mean[key] = var[key] = []

for key, value in data_dict.items():
    data_dict[key]['to_frac'] = getFrac(value['from_poi_to_this_person'],value['to_messages'])
    data_dict[key]['from_frac'] = getFrac(value['from_this_person_to_poi'],value['from_messages'])
    for j in features_list[1:]:
        if value[j] != 'NaN':
            na_count[j].append(value[j])
#print na_count['to_frac']
#print type(na_count['not']['bonus'][0])
#print data_dict['METTS MARK']

for key in features_list[1:]:
    mean[key] = np.mean(na_count[key])
    var[key] = np.var(na_count[key])
#print mean
#print var

features_list.remove('from_poi_to_this_person')
features_list.remove('to_messages')
features_list.remove('from_this_person_to_poi')
features_list.remove('from_messages')
print features_list

data_modf = {}
for key, value in data_dict.items():
    for i in features_list:
        if data_modf[item][key] == 'NaN':
            data_modf[item][key] = mean[key]

'''
for i in features_list[1:7]:
    for j in features_list[7:14]:
        if i == j:
            continue
        plt.scatter(na_count['poi'][i], na_count['poi'][j])
        plt.scatter(na_count['not'][i], na_count['not'][j])
'''

### Task 2: Remove outliers
data = featureFormat(data_dict, features_list)

print data.shape
print type(data)



#for key in data_dict.keys():


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print len(features)

print "label shape is {0}".format(len(labels))
print "features shape is {0}".format(len(features[0]))

print "labels \n",labels
print "# of poi is : {0}".format(sum(labels))
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.svm import SVC as svc

clf1 = gnb()
clf2 = lr()
clf3 = rfc()
clf4 = abc()
clf5 = gbc()
clf6 = svc()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn import metrics as mtr
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

'''
clf1.fit(features_train, labels_train)
print "GNB score is : ", clf1.score(features_test, labels_test)
pred1 = clf1.predict(features_test)
print "GNB precision is : ", mtr.precision_score(labels_test, pred1)
print "GNB recall is : ", mtr.recall_score(labels_test, pred1)

clf2.fit(features_train, labels_train)
print "LR score is : ", clf2.score(features_test, labels_test)
pred2 = clf2.predict(features_test)
print "LR precision is : ", mtr.precision_score(labels_test, pred2)
print "LR recall is : ", mtr.recall_score(labels_test, pred2)

clf3.fit(features_train, labels_train)
print "RF score is : ", clf3.score(features_test, labels_test)
pred3 = clf3.predict(features_test)
print "RF precision is : ", mtr.precision_score(labels_test, pred3)
print "RF recall is : ", mtr.recall_score(labels_test, pred3)

clf4.fit(features_train, labels_train)
print "AB score is : ", clf4.score(features_test, labels_test)
pred4 = clf4.predict(features_test)
print "AB precision is : ", mtr.precision_score(labels_test, pred4)
print "AB recall is : ", mtr.recall_score(labels_test, pred4)

clf5.fit(features_train, labels_train)
print "GB score is : ", clf5.score(features_test, labels_test)
pred5 = clf5.predict(features_test)
print "GB precision is : ", mtr.precision_score(labels_test, pred5)
print "GB recall is : ", mtr.recall_score(labels_test, pred5)

clf6.fit(features_train, labels_train)
print "SVM score is : ", clf6.score(features_test, labels_test)
pred6 = clf6.predict(features_test)
print "SVM precision is : ", mtr.precision_score(labels_test, pred6)
print "SVM recall is : ", mtr.recall_score(labels_test, pred6)
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)