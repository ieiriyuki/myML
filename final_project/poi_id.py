#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from my_defs import getFrac

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'to_messages', 'total_payments', 'bonus', 'shared_receipt_with_poi', 'total_stock_value', 'expenses', 'from_messages', 'from_this_person_to_poi', 'from_poi_to_this_person'] # You will need to use more features
#print("len of features are {0}".format(len(features_list)))

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL')
#print len(data_dict['METTS MARK'])

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# create two new features
features_list.append('to_frac')
features_list.append('from_frac')

na_count, mean, var, sd = {}, {}, {}, {}
for key in features_list[1:]:
    na_count[key] = []
    mean[key] = var[key] = sd[key] = 0

for key, value in my_dataset.items():
    my_dataset[key]['to_frac'] = getFrac(value['from_poi_to_this_person'],value['to_messages'])
    my_dataset[key]['from_frac'] = getFrac(value['from_this_person_to_poi'],value['from_messages'])
    for j in features_list[1:]:
        if value[j] != 'NaN':
            na_count[j].append(value[j])
#print na_count['to_frac']
#print my_dataset['METTS MARK']

#calculate mean values of features
for key in features_list[1:]:
#    print na_count[key]
    mean[key] = np.mean(na_count[key])
    var[key] = np.var(na_count[key])
    sd[key] = np.sqrt(var[key])

features_list.remove('from_poi_to_this_person')
features_list.remove('to_messages')
features_list.remove('from_this_person_to_poi')
features_list.remove('from_messages')
#print("len of features are {0}".format(len(features_list)))

#print "len of dict", len(my_dataset)
# this step is fillin NaN with mean value; well done
for i in my_dataset.keys():
    for j in features_list[1:]:
        if my_dataset[i][j] == 'NaN':
            my_dataset[i][j] = mean[j]

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "label len is {0}".format(len(labels))
print "features len is {0}".format(len(features))
print "# of poi is : {0}".format(sum(labels))

scaler = StandardScaler()
scaler.fit(features)
#print scaler.mean_
#print np.mean(scaler.transform(features), axis=0)
#print np.var(scaler.transform(features), axis=0)
my__features = scaler.transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB as gnb
#from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import AdaBoostClassifier as abc
from sklearn.ensemble import GradientBoostingClassifier as gbc
#from sklearn.svm import SVC as svc

clf1 = gnb()
#clf2 = lr()
clf3 = rfc()
clf4 = abc()
clf5 = gbc()
#clf6 = svc()

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

clf1.fit(features_train, labels_train)
#print "GNB score is : ", clf1.score(features_test, labels_test)
#pred1 = clf1.predict(features_test)
#print "GNB precision is : ", mtr.precision_score(labels_test, pred1)
#print "GNB recall is : ", mtr.recall_score(labels_test, pred1)
test_classifier(clf1, my_dataset, features_list)

#clf2.fit(features_train, labels_train)
#print "LR score is : ", clf2.score(features_test, labels_test)
#pred2 = clf2.predict(features_test)
#print "LR precision is : ", mtr.precision_score(labels_test, pred2)
#print "LR recall is : ", mtr.recall_score(labels_test, pred2)
#test_classifier(clf2, my_dataset, features_list)

clf3.fit(features_train, labels_train)
#print "RF score is : ", clf3.score(features_test, labels_test)
#pred3 = clf3.predict(features_test)
#print "RF precision is : ", mtr.precision_score(labels_test, pred3)
#print "RF recall is : ", mtr.recall_score(labels_test, pred3)
test_classifier(clf3, my_dataset, features_list)

clf4.fit(features_train, labels_train)
#print "AB score is : ", clf4.score(features_test, labels_test)
#pred4 = clf4.predict(features_test)
#print "AB precision is : ", mtr.precision_score(labels_test, pred4)
#print "AB recall is : ", mtr.recall_score(labels_test, pred4)
test_classifier(clf4, my_dataset, features_list)

clf5.fit(features_train, labels_train)
#print "GB score is : ", clf5.score(features_test, labels_test)
#pred5 = clf5.predict(features_test)
#print "GB precision is : ", mtr.precision_score(labels_test, pred5)
#print "GB recall is : ", mtr.recall_score(labels_test, pred5)
test_classifier(clf5, my_dataset, features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)