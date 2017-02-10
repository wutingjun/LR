#coding=utf-8

from numpy import *
import matplotlib.pyplot as plt
import time

from featureScaling import featureNorm
import logRegression
from loadData import loadTrain


# step 1: load data
print "step 1: load data..."
train_x, train_y =loadTrain.loadData('testSet.txt')
numSamples,numFeatures=train_x.shape

train_x = featureNorm.meanNorm(train_x)
train_x=c_[ones([numSamples,1]),train_x]
test_x=train_x;test_y=train_y


# step 2: training...
print "step 2: training..."
opts = {'alpha': 0.01, 'maxIter': 200, 'optimizeType': 'gradDescent'}
optimalWeights = logRegression.trainLogRegres(train_x, train_y, opts)
print optimalWeights

## step 3: testing
print "step 3: testing..."
accuracy = logRegression.testLogRegres(optimalWeights, test_x, test_y)

## step 4: show the result
print "step 4: show the result..."
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
logRegression.showLogRegres(optimalWeights, train_x, train_y)