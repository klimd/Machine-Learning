#!/usr/bin/python

# Klimentina Krstevska
#	2/9.2018

import sys
import os
import csv
import numpy
import random
import math

def calculateRMSE(samples):

	RMSE = 0
	for pair in samples:
		RMSE += math.pow(pair[0]-pair[1], 2)
		
	RMSE = float(RMSE)/len(samples)
	RMSE = math.sqrt(RMSE)
	
	print "RMSE: ", RMSE

def closedForm(trainingData, testingData):

	#Standardize the training data
	trainMean, trainStd, data = standardizeData(trainingData)
	
	#Add an additional feature with value 1 
	for i in range(len(data)):
		row = [1] + data[i]
		data[i] = row
	
	#Compute closed-form solution of linear regression	
	X = []
	Y = []
	for i in range(len(data)):
		X.append(data[i][:-1])
		Y.append([data[i][-1]])
	X = numpy.asmatrix(X)
	Y = numpy.asmatrix(Y)
	
	theta = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(numpy.transpose(X), X)), numpy.transpose(X)), Y)

	#standardize testing set
	testingData = standardizeTestData(testingData, trainMean, trainStd)
	theta =  numpy.squeeze(numpy.asarray(theta))
	print "Theta coefficients: ", theta
	
	#Add an additional feature with value 1 
	for i in range(len(testingData)):
		row = [1] + testingData[i]
		testingData[i] = row
		
	
	samples = []
	#Compute predicted values as xTheta
	for x in testingData:
	
		actual = x[-1]
		sum = 0
		for i in range(len(theta)):
			sum += x[i]*theta[i]
		predicted = sum
		samples.append([actual, predicted])
	
	calculateRMSE(samples)

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

		mean.append(numpy.mean(tempCol))
		std.append(numpy.std(tempCol)) 
		
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
		#
		# 	read in the data
		#
		data = []
		for row in reader:
			r = row[1:]
			data.append(r);
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
	
	closedForm(trainingData, testingData)

def main():

	if len(sys.argv) != 2:
		print 'Please specify a filename'
	
	else:
		filename = sys.argv[1];
		parseData(filename)

main()