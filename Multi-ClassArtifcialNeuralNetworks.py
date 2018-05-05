#!/usr/bin/python
#
# Klimentina Krstevska
#	3/16/2018
import matplotlib
matplotlib.use('Agg')


import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sets import Set

def findK(Y):

	k = set()
	for y in Y:
		k.add(y)
	return k
	
def activationFunction(x):
	
	return float(1)/(1 + math.pow(math.e, -x))
	
def calculateAccuracy(Y, O):

	match = 0
	for i in xrange(len(Y)):
		if(int(Y[i]) == int(O[i])):
			match += 1
	accuracy = float(match)/len(Y) * 100
	
	return round(accuracy,2)

def mcann(trainingData, testingData, M):

	#	standardize the data
	mean, std, stanData, k = standardizeData(trainingData)
	standTestData = standardizeTestData(testingData, mean, std)

	D = len(stanData[0]) - 2 # number of features
	k = list(k)
	K = len(k)
	# Randomly initialize weights from input layer to hidden layers, beta of size D + 1 x M
	beta = []
	for i in xrange(int(D+1)):
		temp = []
		for j in xrange(int(M)):
			temp.append(random.uniform(-1, 1))
		beta.append(temp)
	beta = np.matrix(beta)	
	
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
	Yerr = []
	for row in stanData:
		X.append(row[:-1]) 	#features
		Y.append([float(row[-1])])	#class id
		temp = []
		for i in xrange(K):
			temp.append(float(row[-1]))	
		Yerr.append(temp)
	X = np.asarray(X)
	Y = np.asarray(Y)
	Yerr = np.asarray(Yerr)
	
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
		Opredicted = []
		for row in output:
			temp = []
			for f in row:			
				temp.append(activationFunction(f))	
			O.append(temp)
			maxProb = max(temp)
			index = temp.index(maxProb)
			classF = k[index]
			Opredicted.append([classF])
		O = np.asarray(O)	
		Opredicted = np.asarray(Opredicted);
		
		#compute the error at the output layer
		d_out = []
		for i in xrange(len(Yerr)):
			d_out.append(Yerr[i] - O[i])
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
		trainingAccuracy = calculateAccuracy(Y, Opredicted)
		TA.append(trainingAccuracy)
		
		iter +=1
		
	plt.plot(TA)	
	plt.xlabel("Iterations")
	plt.ylabel("Training Accuracy")
	plt.savefig("mcann.png")		
		
	#
	#test the data
	#
	
	Xtest = []
	Ytest = []
	for row in standTestData:
		Xtest.append(row[:-1]) 	#features
		Ytest.append([float(row[-1])])	#class id
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
	for row in output:
		temp = []
		for f in row:			
			temp.append(activationFunction(f))
		maxProb = max(temp)
		index = temp.index(maxProb)
		classF = k[index]
		Otest.append([classF])
	Otest = np.asarray(Otest)	
	testingAccuracy = calculateAccuracy(Ytest, Otest)
	print "The testing accuracy is: ", testingAccuracy	
	
def standardizeTestData(data, mean, std):	

	stanData = []
	for row in data:
		tempCol = [row[0]]
		for i in range(1, len(row)-2):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		stanData.append(tempCol)
		
	return stanData
	
def standardizeData(data):

	#
	#	calculate mean and standard deviation for each column
	#
	mean = [0]
	std = [0]
	for i in xrange(1,len(data[0])-2):
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
		tempCol = [row[0]]
		for i in range(1,len(row)-2):
			tempCol.append((float(row[i]) - mean[i])/std[i])
		tempCol.append(float(row[len(row)-1]))
		Y.append(float(row[len(row)-1]))
		stanData.append(tempCol)
	
	k = findK(Y)	
		
	return mean, std, stanData, k


def parseData(filename, M):

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
	
	mcann(trainingData, testingData, M)

def main():

	if len(sys.argv) != 3:
		print 'Please specify a filename and M'
	
	else:
		filename = sys.argv[1]
		M = sys.argv[2]
		parseData(filename, M)

main()