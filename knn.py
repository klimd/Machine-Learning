#!/usr/bin/python

# Klimentina Krstevska
#	2/23/2018

import sys
import os
import csv
import numpy as np
import random
import math
	
def calculateStatistics(testingData, predictedClass):

	actualClass = [] 
	for row in testingData:
		actualClass.append(int(row[-1]))
		
	TP = 0 ; TN = 0 ; FP = 0 ; FN = 0
	for i in range(len(actualClass)):
	
		# Positive examples
		if actualClass[i] == 1:
			# Positive example and predicted positive
			if predictedClass[i] == 1:
				TP += 1
			# Positive example and predicted negative
			else :
				FN += 1
				
		# Negative examples	
		if actualClass[i] == 0:
			# Negative example and predicted positive
			if predictedClass[i] == 1:
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
	
	print "Precision: ", precision
	print "Recall: ", recall
	print "F-measure: ", fmeasure
	print "Accuracy: ", accuracy
	
	
def knn(trainingData, testingData, k):

	#
	#	standardize the data
	#
	mean, std, stanData = standardizeData(trainingData)
	standTestData = standardizeTestData(testingData, mean, std)
	
	#vfunc = np.vectorize(manhattanDist)
	predictedClass = {}
	i = 0
	for x in standTestData:
		neighDis = []
		
		#compute the Manhattan distance
		for xi in stanData:
			d = manhattanDist(xi, x)
			c = int(xi[-1])
			neighDis.append((d,c))
			
		# find k nearest neighbours	
		neighDis.sort(key=lambda tup: tup[0])	
		neighDis = neighDis[:int(k)]
		
		#find the class of the neighbours
		modes = {}
		for tup in neighDis:
			if tup[1] in modes.keys():
				modes[tup[1]] += 1
			else:
				modes[tup[1]] = 1
				
		#find the majority class(es)
		sortedModes = sorted(modes.items(), key= lambda (k,v): (v,k), reverse=True)
		maxMode = sortedModes[0][1]
		for pair in sortedModes:
			if pair[1] != maxMode:
				del modes[pair[0]]
				
		#now the modes dictionary only has the values with the maximum mode
		# we pick a random one of the max modes
		# if there is only one, that one will always be the one to get picked
		Class = random.choice(modes.keys())
		
		predictedClass[i] = Class
		i+=1
	
	calculateStatistics(testingData, predictedClass.values())
	
def manhattanDist (xi, x):

	d = 0
	for i in xrange(len(xi)-1):
		d += abs(xi[i] - x[i])
	
	return d	

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


def parseData(filename, k):

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
	
	knn(trainingData, testingData, k)

def main():

	if len(sys.argv) != 3:
		print 'Please specify a filename and k'
	
	else:
		filename = sys.argv[1];
		k = sys.argv[2]
		parseData(filename, k)

main()
