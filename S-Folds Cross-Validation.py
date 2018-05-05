#!/usr/bin/python

import matplotlib
matplotlib.use('Agg')

import sys
import os
import matplotlib.pyplot as plt
import csv
import numpy
import random
import math

error = []

def calculateError(samples):

	for pair in samples:
		error.append(math.pow(pair[0]-pair[1], 2))
	
def calculateRMSE(error):

	RMSE = 0
	for err in error:
		RMSE += err
		
	RMSE = float(RMSE)/len(error)
	RMSE = math.sqrt(RMSE)
	
	print RMSE

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
	#print theta
	
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
			#print i, x[i], theta[i]
			sum += x[i]*theta[i]
		predicted = sum
		samples.append([actual, predicted])
	
	calculateError(samples)

def standardizeTestData(data, mean, std):	

	stanData = []
	for row in data:
		tempCol = []
		for i in range(len(row)-1):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	return stanData
	
def sFolds(folds):

	initial = folds
	for i in range(len(folds)):
		testingData = folds[0]
		del folds[0]
		trainingData = folds
		
		temp = []
		for fold in trainingData:
			for row in fold:
				temp.append(row)
				
		#standardize the data
		closedForm(temp, testingData)
		
		trainingData.append(testingData)
	
	#compute RMSE
	calculateRMSE(error)
	
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


def parseData(filename, S):

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
	#	separate data into S Folds
	#		
	x = len(data)/S
	k = 0
	folds = []
	temp = []
	for i in range(len(data)):
		if k <= x:
			temp.append(data[i])
			k+=1
		else:
			folds.append(temp)
			temp = []
			temp.append(data[i])
			k = 1
	folds.append(temp)
	
	sFolds(folds)	

def main():

	if len(sys.argv) != 3:
		print 'Please specify a filename and S'
	
	else:
		filename = sys.argv[1];
		S = int(sys.argv[2])
		parseData(filename, S)

main()