#coding=utf-8
from numpy import *

def loadData(filename):
	train_x = []
	train_y = []
	fileIn = open(filename)
	for line in fileIn:
		lineArr = line.strip().split('\t')
		train_x.append([float(lineArr[0]), float(lineArr[1])])
		train_y.append(float(lineArr[2]))
	fileIn.close()
	return mat(train_x), mat(train_y).transpose()