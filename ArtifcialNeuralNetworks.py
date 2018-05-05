#!/usr/bin/python

# Klimentina Krstevska
#	3/15/2018

import matplotlib
matplotlib.use('Agg')

import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
	
def calculateAccuracy(Y, O):

	match = 0
	for i in xrange(len(Y)):
		if(int(Y[i]) == int(O[i])):
			match += 1
	accuracy = 1.0/len(Y) * match * 100
	
	return round(accuracy, 2)

def findK(Y):

	k = set()
	for y in Y:
		k.add(y)
	return k
	
def activationFunction(x):
	
	return float(1)/(1 + math.pow(math.e, -x))
	
def ann(trainingData, testingData, M):

	#	standardize the data
	mean, std, stanData, K = standardizeData(trainingData)
	standTestData = standardizeTestData(testingData, mean, std)
	
	D = len(trainingData[0]) - 2 # number of features
	
	# Randomly initialize weights from input layer to hidden layers, beta of size D + 1 x M
	beta = []
	for i in xrange(int(D+1)):
		temp = []
		for j in xrange(int(M)):
			temp.append(random.uniform(-1, 1))
		beta.append(temp)
	beta = np.matrix(beta)	
	
	K = 1
	# Randomly initialize weights from hidden layer to output layer, theta of size M x K
	theta = []
	for i in xrange(int(M)):
		temp = []
		for j in xrange(int(K)):
			temp.append(random.uniform(-1, 1))
		theta.append(temp)
	theta = np.asarray(theta)	
		
	n = 0.5	
	N = len(trainingData)	
	const = (float(n)/N)	
	iter = 1
	iterations = 1000
	
	X = []
	Y = []
	for row in stanData:
		X.append(row[:-1]) 	#features
		Y.append([float(row[-1])])	#class id
	X = np.asarray(X)
	Y = np.asarray(Y)

	TA =[]	#training accuracy	
	while iter <= iterations:
		
		# H = g(X * beta)
		input = np.dot(X, beta)
		input = np.asarray(input)
		H = []
		for row in input:
			temp = []
			for f in row:			
				temp.append(activationFunction(f))
			H.append(temp)
		H = np.asarray(H)
		
		# O = g(H * theta)		
		output = np.dot(H, theta)
		output = np.asarray(output)
		O = []
		for o in output:
			predictedVal = activationFunction(o)
			if predictedVal > 0.5 :
				o = 1.0
			else:
				o = 0.0
			O.append([float(o)])
		O = np.asarray(O)	
		
		#compute the error at the output layer
		d_out = []
		for i in xrange(len(Y)):
			d_out.append(Y[i] - O[i])
		d_out = np.asarray(d_out)
		d_out = np.asarray(d_out)	
			
		#Update the weights from hidden layer to output layer
		HT = np.transpose(np.matrix(H))	
		temp =  const * HT * d_out
		thetanew = np.matrix(theta) + np.matrix(temp)
		theta = thetanew
		
		#compute the error at the hidden layer 
		thetaT = np.transpose(np.matrix(theta))
		H1 = np.asarray(1-H)
		d_hidden = np.multiply(np.multiply(d_out * thetaT , H), H1)

		#Update the weights from input layer to hidden layer
		XT = np.transpose(np.matrix(X))
		temp1 = const * XT * d_hidden
		betanew = np.matrix(beta) + np.matrix(temp1)	
		beta = betanew
		
		#calculate accuracy for this iteration
		trainingAccuracy = calculateAccuracy(Y, O)
		TA.append(trainingAccuracy)
	
		iter += 1
	
	plt.plot(TA)	
	plt.xlabel("Iterations")
	plt.ylabel("Training Accuracy")
	plt.savefig("ann.png")		
	
	#
	#test the data
	#
	
	Xtest = []
	Ytest = []
	for row in standTestData:
		Xtest.append(row[:-1])
		Ytest.append(row[-1])
	Xtest = np.asarray(Xtest)
	Ytest = np.asarray(Ytest)
	
	# H = g(X * beta)
	input = np.dot(Xtest, beta)
	input = np.asarray(input)
	Htest = []
	for row in input:
		temp = []
		for f in row:
			temp.append(activationFunction(f))
		Htest.append(temp)
	Htest = np.asarray(Htest)
		
	# O = g(H * theta)		
	output = np.dot(Htest, theta)
	output = np.asarray(output)
	Otest = []
	for o in output:
		predictedVal = activationFunction(o)
		if predictedVal > 0.5 :
			o = 1
		else:
			o = 0
		Otest.append(float(o))
	Otest = np.asarray(Otest)	
	testingAccuracy = calculateAccuracy(Ytest, Otest)
	print "The testing accuracy is: ", testingAccuracy
	
	
def standardizeTestData(data, mean, std):	

	stanData = []
	for row in data:
		tempCol = [row[0]]
		for i in range(1, len(row)-1):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	return stanData
	
def standardizeData(data):

	#
	#	calculate mean and standard deviation for the 
	#	features columns (1 to 57)
	#
	mean = [0]
	std = [0]
	for i in xrange(1,len(data[0])-1):
		tempCol = []
		for row in data:
			tempCol.append(float(row[i]))

		mean.append(np.mean(tempCol))
		std.append(np.std(tempCol)) 
		
	#
	#	standardize the data
	#
	stanData = []
	Y = []
	for row in data:
		tempCol = [row[0]]
		for i in range(1,len(row)-1):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		Y.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	k = findK(Y)
	
	return mean, std, stanData, k


def parseData(filename, M):

	with open(filename, 'rb') as f:
		reader = csv.reader(f)	
		colNum = len(next(reader))
		f.seek(0)
		#
		# 	read in the data
		#
		data = []
		for row in reader:
			#add biased feature
			row = [1.0] + row
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
	
	ann(trainingData, testingData, M)

def main():

	if len(sys.argv) != 3:
		print 'Please specify a filename and M'
	
	else:
		filename = sys.argv[1]
		M = sys.argv[2]
		parseData(filename, M)

main()
