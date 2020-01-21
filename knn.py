# user defined functions KNN

# packages
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data
training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")
train_row = 42000
Origin_X_train = training.values[0:train_row,1:]   # pixel0 - pixel783
Origin_y_train = training.values[0:train_row, 0]   # label
Origin_X_test = testing.values  # testing data without labels

X_train,X_vali, y_train, y_vali = train_test_split(Origin_X_train,
                                                   Origin_y_train,
                                                   test_size = 0.2,
                                                   random_state = 0)

# KNN functions
class MyKnn():
    
    def __init__(self):
        pass # here we do not need init
    
    def train(self, X, y):
        # the training part of KNN is easy, just the previous information
        # we'll predict the results based on such information
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=3):
        # default: the number of neighbours is 3
        dataSet = X_train
        labels = y_train
        # The number of records
        dataSetSize = dataSet.shape[0]
        
        # calculate the distances by Matrix
        diffMat = np.tile(X, [dataSetSize, 1]) - dataSet
        sqDiffMat = diffMat ** 2
        sumDiffMat = sqDiffMat.sum(axis=1)
        distances = sumDiffMat ** 0.5
        
        # sort and count the labels
        sortedDistances = distances.argsort()
        # count how many instances for each digit number appear in KNN
        # we use a hash table to store label information
        classCount = {}
        for i in range(k):
            vote = labels[sortedDistances[i]]
            classCount[vote] = classCount.get(vote, 0) + 1
        # find the label corresponding to the maximum count
        max_count = 0
        ans = 0
        for key, val in classCount.items():
            if val > max_count:
                ans = key
                max_count = val
                
        return ans


# KNN training
classifier = MyKnn()
classifier.train(X_train, y_train)

max = 0
ans_k = 0 
for k in range(1,4):
    print("k=", k, "starting validation")
    predictions = np.zeros(len(y_vali))
    for i in range(X_vali.shape[0]):
        if i % 500 == 0: 
            print(i)
        output =  classifier.predict(X_vali[i], k) 
        predictions[i] = output
        
    accuracy = accuracy_score(y_vali, predictions) 
    print("k=", k, "accuracy=", accuracy)
    
    if max < accuracy:
        max = accuracy
        ans_k = k



