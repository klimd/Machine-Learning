#!/usr/bin/python

# Klimentina Krstevska
#	2/28/2018

import sys
import os
import csv
import numpy as np
import random
import math
from sklearn import svm
	
def calculateStatistics(testingData, predictedClass):

	TP = 0 ; TN = 0 ; FP = 0 ; FN = 0
	for i in range(len(testingData)):
		#print int(testingData[i]), int(predictedClass[i])
		# Positive examples
		if int(testingData[i]) == 1:
			# Positive example and predicted positive
			if int(predictedClass[i]) == 1:
				TP += 1
			# Positive example and predicted negative
			else :
				FN += 1
				
		# Negative examples	
		if int(testingData[i]) == 0:
			# Negative example and predicted positive
			if int(predictedClass[i]) == 1:
				FP +=1
			# Negative example and predicted negative
			else :
				TN += 1
	
	# calculate precision
	precision = float(TP) /(TP + FP) * 100
	
	# calculate recall
	recall = float(TP) /(TP + FN) * 100
	
	# calculate fmeasure
	fmeasure = (2 * precision * recall)/(precision + recall)
	
	# calculate accuracy 
	accuracy = float(TP + TN)/(TP + TN + FP + FN) * 100
	
	print "Precision: ", round(precision, 2), "%"
	print "Recall: ", round(recall, 2), "%"
	print "F-measure: ", round(fmeasure, 2), "%"
	print "Accuracy: ", round(accuracy, 2), "%"
	
	
def svmFunction(trainingData, testingData):

	#
	#	standardize the data
	#
	mean, std, stanData = standardizeData(trainingData)
	standTestData = standardizeTestData(testingData, mean, std)
	
	#
	# For the training data:
	# Get the features in X and the classifications in Y
	#
	X = []
	Y = []
	for row in stanData:
		temp = []
		for i in xrange(0, len(row) - 1):
			temp.append(row[i])
		
		X.append(temp)
		Y.append(row[len(row)-1])
				
	#
	# For the testing data:
	# Get the features in testX and the classifications in testY
	#	
	testX = []
	testY = []
	for row in standTestData:
		temp = []
		for i in xrange(0, len(row) - 1):
			temp.append(row[i])
		
		testX.append(temp)
		testY.append(row[len(row)-1]) 
	
	#
	# Train SVM on the training data 
	#
	clf = svm.SVC()
	clf.fit(X, Y)	
	
	# Classify the SVM using the testing set
	predictedValues = clf.predict(testX)	
	
	#Calculate the statistics 
	calculateStatistics(testY, predictedValues)
	

def standardizeTestData(data, mean, std):	

	stanData = []
	for row in data:
		tempCol = []
		for i in range(len(row)-1):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	return stanData
	
def standardizeData(data):

	#
	#	calculate mean and standard deviation for each column
	#
	mean = []
	std = []
	for i in xrange(len(data[0])-1):
		tempCol = []
		for row in data:
			tempCol.append(float(row[i]))

		mean.append(np.mean(tempCol))
		std.append(np.std(tempCol)) 
		
	#
	#	standardize the data
	#
	stanData = []
	for row in data:
		tempCol = []
		for i in range(len(row)-1):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	return mean, std, stanData


def parseData(filename):

	with open(filename, 'rb') as f:
		reader = csv.reader(f)	
		colNum = len(next(reader))
		f.seek(0)
		#
		# 	read in the data
		#
		data = []
		for row in reader:
			data.append(row);
	#
	#	randomize the data
	#
	random.seed(0)
	random.shuffle(data)
	
	#
	#	separate training and testing data
	#
	x =  int( (float(2)/3) * len(data))
	trainingData = []
	testingData = []
	
	for i in range(len(data)):
		if i <= x:
			trainingData.append(data[i])
		else:
			testingData.append(data[i])
	
	svmFunction(trainingData, testingData)

def main():

	if len(sys.argv) != 2:
		print 'Please specify a filename'
	
	else:
		filename = sys.argv[1]
		parseData(filename)

main()
