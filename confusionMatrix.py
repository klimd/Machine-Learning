#!/usr/bin/python
#
# Klimentina Krstevska
#	3/2/2018

import sys
import os
import csv
import numpy as np
import random
import math
from sklearn import svm
from sets import Set

def findK(Y):

	k = set()
	for y in Y:
		k.add(y)
	return k

def calculateConfusionMatrix(testY, predictedLabels, k):

	# Creating the initial confusion market
	# 0 .. .. 0
	# ..     ..
	# 0 .. .. 0
	confusionMatrix = []
	for i in xrange(len(k)):
		temp = []
		for j in xrange(len(k)):
			temp.append(0)
		confusionMatrix.append(temp)
	
	labels = list(k)
	for i in xrange(len(testY)):
		for label in labels:
			if(testY[i] == label):
				n = labels.index(testY[i])
			if(predictedLabels[i] == label):
				m = labels.index(predictedLabels[i])
				
		confusionMatrix[n][m] += 1
	
	size = len(testY)
	for i in xrange(len(k)):
		for j in xrange(len(k)):
			confusionMatrix[i][j] = round((float(confusionMatrix[i][j])/size)*100, 2)
		
	print confusionMatrix
		
def svmMulti(trainingData, testingData):

	#
	#	standardize the data
	#
	mean, std, stanData, k = standardizeData(trainingData)
	standTestData = standardizeTestData(testingData, mean, std)

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
	# separate the training data according to the class it belongs to
	#
	classSeparatedData = {}
	for row in stanData:
		key = row[-1]
		features = row[:-1]
		if key in classSeparatedData.keys():
			temp = classSeparatedData[key]
			temp.append(features)
			classSeparatedData[key] = temp
		else:
			classSeparatedData[key] = [features]

	#		
	# Train using One vs One
	#
	classes = classSeparatedData.keys(); 
	classifiers = []
	for i in xrange(len(classes)):
		for j in xrange(i+1, len(classes)):
			
			class1 = classes[i]
			class2 = classes[j]
			
		# For the training data:
		# Get the features in X and the classifications in Y
			X = []
			Y = []
			for row in classSeparatedData[class1]:
				X.append(row)
				Y.append(class1)
			for row in classSeparatedData[class2]:
				X.append(row)
				Y.append(class2)
						
			#
			# Train SVM on the training data 
			#
			clf = svm.SVC()
			clf.fit(X, Y)	
			
			# Classify SVM using the testing set
			predictedValues = clf.predict(testX)
			classifiers.append(predictedValues)

	#
	# pick a label for the test observations
	# 
	predictedLabels = []
	for i in xrange(len(testY)):
		temp = []		
		# get all the predicted values for the sample
		for j in xrange(len(classifiers)):
			temp.append(classifiers[j][i])
		
		# find the mod of the predicted values
		modPredicted = {}
		for x in temp:
			if x in modPredicted.keys():
				modPredicted[x] += 1
			else:
				modPredicted[x] = 1
		
		#find the majority class(es)
		sortedModes =  sorted(modPredicted.iteritems(), key=lambda (k,v): (v,k), reverse=True)
		maxMode = sortedModes[0][1]
		for pair in sortedModes:
			if pair[1] != maxMode:
				del modPredicted[pair[0]]
				
				
		#now the modes dictionary only has the values with the maximum mode
		# we pick a random one of the max modes
		# if there is only one, that one will always be the one to get picked
		predictedY = random.choice(modPredicted.keys())		
		
		predictedLabels.append(predictedY)

	#Calculate the statistics 
	calculateConfusionMatrix(testY, predictedLabels, k)

	
def standardizeTestData(data, mean, std):	

	stanData = []
	for row in data:
		tempCol = []
		for i in range(len(row)-2):
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
	for i in xrange(len(data[0])-2):
		tempCol = []
		for row in data:
			tempCol.append(float(row[i]))

		mean.append(np.mean(tempCol))
		std.append(np.std(tempCol)) 
		
	#
	#	standardize the data
	#
	Y = []
	stanData = []
	for row in data:
		tempCol = []
		for i in range(len(row)-2):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		Y.append(float(row[len(row)-1]))
		stanData.append(tempCol)
	
	k = findK(Y)
	
	return mean, std, stanData, k 


def parseData(filename):

	with open(filename, 'rb') as f:
		reader = csv.reader(f)	
		
		# read the first two rows
		colNum = len(next(reader))
		colNum = len(next(reader))
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
	
	svmMulti(trainingData, testingData)

def main():

	if len(sys.argv) != 2:
		print 'Please specify a filename'
	
	else:
		filename = sys.argv[1]
		parseData(filename)

main()